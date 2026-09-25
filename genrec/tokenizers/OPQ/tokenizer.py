"""OPQ SID tokenizer: sentence emb → PCA → FAISS OPQ/PQ.

SIDs are ``N_DIGIT`` codes of ``CODEBOOK_SIZE`` each; sentence embeddings
are PCA-reduced to ``SENT_EMB_PCA`` dims and L2-normalised before OPQ.

Exposes the SPRINT MultiHeadVQVAE-compatible surface:
``item2tokens``, ``n_digit``, ``codebook_sizes``, ``eos_token``, ``ignored_label``.

Token layout:
  PAD=0, BOS=1, EOS=2, SID start at ``sid_offset=3``.
MASK for SPRINT mean-flow sits at ``mask_token_id = sid_offset + n_digit * K``.
"""

from __future__ import annotations

import json
import math
import os
import pickle
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer

from genrec.constants import CODEBOOK_SIZE, N_DIGIT, SENT_EMB_PCA
from genrec.dataset import AbstractDataset
from genrec.tokenizer import AbstractTokenizer


class OPQTokenizer(AbstractTokenizer):

    def __init__(self, config: dict, dataset: AbstractDataset):
        config.setdefault('device',
                          'cuda' if __import__('torch').cuda.is_available() else
                          'cpu')
        config.setdefault('num_proc', 1)
        super().__init__(config, dataset)

        self.n_codebook_bits = self._get_codebook_bits(CODEBOOK_SIZE)
        if config.get('disable_opq', False):
            self.index_factory = (
                f'IVF1,PQ{N_DIGIT}x{self.n_codebook_bits}')
        else:
            self.index_factory = (
                f'OPQ{N_DIGIT},IVF1,PQ{N_DIGIT}x{self.n_codebook_bits}')

        self.log(f'[TOKENIZER] OPQ index factory: {self.index_factory}')
        self.dataset = dataset
        self.item2id = dataset.item2id
        self.id2item = dataset.id_mapping['id2item']

        self.pad_token = 0
        self.bos_token = 1
        self.eos_token = 2
        self.sid_offset = 3
        self.ignored_label = -100
        # After all SID slices so SPRINT can put [MASK] in the embedding table.
        self.mask_token_id = (self.sid_offset +
                              self.n_digit * self.codebook_size)

        self.item2tokens = self._init_tokenizer(dataset)
        if not hasattr(self, 'tokens2item'):
            self.tokens2item = self._create_reverse_mapping()

    @property
    def n_digit(self):
        return N_DIGIT

    @property
    def codebook_size(self):
        return CODEBOOK_SIZE

    @property
    def codebook_sizes(self):
        return [self.codebook_size] * self.n_digit

    @property
    def vocab_size(self) -> int:
        # PAD/BOS/EOS + SID tokens; MASK is +1 beyond this (see mask_token_id).
        return self.sid_offset + self.n_digit * self.codebook_size

    def _get_codebook_bits(self, n_codebook: int) -> int:
        x = math.log2(n_codebook)
        assert x.is_integer() and x >= 0, "Invalid codebook_size"
        return int(x)

    def _cache_dir(self, dataset: AbstractDataset) -> str:
        cache_dir = os.path.join(dataset.cache_dir, 'processed')
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    def _quant_tag(self) -> str:
        return self.index_factory

    def _encode_sent_emb(self, dataset: AbstractDataset,
                         output_path: str) -> np.ndarray:
        meta_sentences = []
        for i in range(1, dataset.n_items):
            meta_sentences.append(
                dataset.item2meta[dataset.id_mapping['id2item'][i]])

        model_id = self.config['sent_emb_model']
        sent_emb_model = SentenceTransformer(
            model_id, trust_remote_code=True).to(self.config['device'])
        sent_embs = sent_emb_model.encode(
            meta_sentences,
            convert_to_numpy=True,
            batch_size=self.config['sent_emb_batch_size'],
            show_progress_bar=True,
            device=self.config['device'],
            normalize_embeddings=True,
        )
        sent_embs.tofile(output_path)
        return sent_embs

    def _get_items_for_training(self, dataset: AbstractDataset) -> np.ndarray:
        items_for_training = set()
        split_data = dataset.split()
        train_dataset = split_data['train']
        if hasattr(train_dataset,
                   'column_names') and 'item_seq' in train_dataset.column_names:
            for item_seq in train_dataset['item_seq']:
                if isinstance(item_seq, (list, tuple)):
                    items_for_training.update(item_seq)
                else:
                    items_for_training.add(item_seq)
        n_sent_embs = dataset.n_items - 1
        self.log(
            f'[TOKENIZER] Items for training: {len(items_for_training)} of '
            f'{n_sent_embs}')
        mask = np.zeros(n_sent_embs, dtype=bool)
        for item in items_for_training:
            item_id = dataset.item2id[item]
            if 1 <= item_id < dataset.n_items:
                mask[item_id - 1] = True
        return mask

    @staticmethod
    def _sid_uniqueness(codes: np.ndarray):
        uniq = len({tuple(int(x) for x in c) for c in codes})
        tot = int(codes.shape[0])
        rate = (tot - uniq) / tot if tot else 0.0
        return tot, uniq, rate

    def _generate_semantic_id_opq(self, sent_embs, sem_ids_path, train_mask):
        import faiss

        self.log(f'[TOKENIZER] sent_embs shape: {sent_embs.shape}')
        self.log(f'[TOKENIZER] train_mask True count: {int(np.sum(train_mask))}')

        if self.config['opq_use_gpu']:
            res = faiss.StandardGpuResources()
            res.setTempMemory(1024 * 1024 * 512)
            co = faiss.GpuClonerOptions()
            co.useFloat16 = self.n_digit >= 56
        faiss.omp_set_num_threads(self.config['faiss_omp_num_threads'])
        index = faiss.index_factory(sent_embs.shape[1], self.index_factory,
                                    faiss.METRIC_INNER_PRODUCT)
        self.log('[TOKENIZER] Training FAISS index...')
        if self.config['opq_use_gpu']:
            index = faiss.index_cpu_to_gpu(res, self.config['opq_gpu_id'],
                                           index, co)
        index.train(sent_embs[train_mask])
        index.add(sent_embs)
        if self.config['opq_use_gpu']:
            index = faiss.index_gpu_to_cpu(index)

        if isinstance(index, faiss.IndexPreTransform):
            ivf_index = faiss.downcast_index(index.index)
        else:
            ivf_index = faiss.downcast_index(index)

        invlists = faiss.extract_index_ivf(ivf_index).invlists
        ls = invlists.list_size(0)
        codes_ptr = invlists.get_codes(0)
        ids_ptr = invlists.get_ids(0)
        pq_codes_u8 = faiss.rev_swig_ptr(codes_ptr, ls * invlists.code_size)
        ids = faiss.rev_swig_ptr(ids_ptr, ls).copy()
        pq_codes_u8 = pq_codes_u8.reshape(-1, invlists.code_size)

        n_items = sent_embs.shape[0]
        codes = np.zeros((n_items, self.n_digit), dtype=np.int64)
        n_bytes = invlists.code_size
        for pos, u8code in enumerate(pq_codes_u8):
            bs = faiss.BitstringReader(faiss.swig_ptr(u8code), n_bytes)
            code = [bs.read(self.n_codebook_bits) for _ in range(self.n_digit)]
            iid0 = int(ids[pos])
            if 0 <= iid0 < n_items:
                codes[iid0] = np.asarray(code, dtype=np.int64)

        tot, uniq, rate = self._sid_uniqueness(codes)
        self.log(f'[TOKENIZER] FAISS SIDs: {uniq}/{tot} unique, '
                 f'collision_rate={rate:.6f} (collisions are kept as-is)')

        indices_count = defaultdict(int)
        for c in codes:
            indices_count['-'.join(str(int(x)) for x in c)] += 1
        self.log(f'[TOKENIZER] Max number of conflicts: '
                 f'{max(indices_count.values()) if indices_count else 0}')

        item2sem_ids = {}
        for i in range(n_items):
            item = self.id2item[i + 1]
            item2sem_ids[item] = tuple(int(v) for v in codes[i])

        self.log(f'[TOKENIZER] Saving semantic IDs to {sem_ids_path}...')
        os.makedirs(os.path.dirname(sem_ids_path) or '.', exist_ok=True)
        with open(sem_ids_path, 'w') as f:
            json.dump(item2sem_ids, f)

    def _sem_ids_to_tokens(self, item2sem_ids: dict) -> dict:
        for item in item2sem_ids:
            tokens = list(item2sem_ids[item])
            tokens = [
                int(t) + self.sid_offset + d * self.codebook_size
                for d, t in enumerate(tokens)
            ]
            item2sem_ids[item] = tuple(tokens)
        return item2sem_ids

    def _sem_ids_filename(self) -> str:
        model_basename = os.path.basename(self.config['sent_emb_model'])
        return (f'{model_basename}_pca{SENT_EMB_PCA}_'
                f'{self._quant_tag()}.sem_ids')

    def _resolve_sem_ids_path(self, dataset: AbstractDataset) -> str:
        """Existing SID file if one is on disk, otherwise where to train one.

        An explicit ``sem_ids_path`` wins. Otherwise look for the ready-made
        file in the category cache (for example
        ``cache/AmazonReviews2014/Beauty/*.sem_ids``) and then in
        ``processed/``. A missing file is trained into the category cache.
        """
        explicit = self.config.get('sem_ids_path')
        if explicit not in (None, '~', ''):
            return explicit

        name = self._sem_ids_filename()
        candidates = [
            os.path.join(dataset.cache_dir, name),
            os.path.join(dataset.cache_dir, 'processed', name),
        ]
        for path in candidates:
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                return path
        return candidates[0]

    def _load_or_build_sent_embs(self, dataset: AbstractDataset,
                                 cache_dir: str):
        model_basename = os.path.basename(self.config['sent_emb_model'])
        raw_path = os.path.join(
            cache_dir,
            f'{model_basename}_raw_d{self.config["sent_emb_dim"]}.sent_emb')
        pca_path = os.path.join(
            cache_dir,
            f'{model_basename}_pca{SENT_EMB_PCA}.sent_emb')
        # Fall back to legacy LLaDA cache name if present.
        legacy_path = os.path.join(cache_dir, f'{model_basename}.sent_emb')

        if os.path.exists(pca_path):
            self.log(
                f'[TOKENIZER] Loading PCA-ed sentence embeddings from '
                f'{pca_path}...')
            return np.fromfile(pca_path, dtype=np.float32).reshape(
                -1, SENT_EMB_PCA)

        if os.path.exists(raw_path):
            self.log(
                f'[TOKENIZER] Loading RAW sentence embeddings from '
                f'{raw_path}...')
            raw_embs = np.fromfile(raw_path, dtype=np.float32).reshape(
                -1, self.config['sent_emb_dim'])
        elif os.path.exists(legacy_path):
            self.log(
                f'[TOKENIZER] Loading legacy sentence embeddings from '
                f'{legacy_path}...')
            raw_embs = np.fromfile(legacy_path, dtype=np.float32).reshape(
                -1, self.config['sent_emb_dim'])
        else:
            self.log('[TOKENIZER] Encoding sentence embeddings...')
            raw_embs = self._encode_sent_emb(dataset, raw_path)

        # PCA fitted on training items, then L2-normalised.
        self.log('[TOKENIZER] Applying PCA to sentence embeddings...')
        from sklearn.decomposition import PCA
        pca = PCA(n_components=SENT_EMB_PCA, whiten=True)
        training_item_mask = self._get_items_for_training(dataset)
        pca.fit(raw_embs[training_item_mask])
        sent_embs = pca.transform(raw_embs).astype(np.float32, copy=False)
        norms = np.linalg.norm(sent_embs, axis=1, keepdims=True) + 1e-12
        sent_embs = sent_embs / norms
        sent_embs.tofile(pca_path)
        return sent_embs

    def _init_tokenizer(self, dataset: AbstractDataset):
        cache_dir = self._cache_dir(dataset)
        sem_ids_path = self._resolve_sem_ids_path(dataset)
        force_regenerate = bool(self.config.get('force_regenerate_opq', False))
        sid_ready = (os.path.isfile(sem_ids_path)
                     and os.path.getsize(sem_ids_path) > 0)

        if sid_ready and not force_regenerate:
            self.log(f'[TOKENIZER] Using existing SIDs from {sem_ids_path}')
        else:
            if force_regenerate:
                self.log(
                    f'[TOKENIZER] Force regenerating SIDs → {sem_ids_path}')
            else:
                self.log(
                    f'[TOKENIZER] SID file not found, training OPQ → '
                    f'{sem_ids_path}')
            sent_embs = self._load_or_build_sent_embs(dataset, cache_dir)
            self.log(
                f'[TOKENIZER] Sentence embeddings shape: {sent_embs.shape}')
            training_item_mask = self._get_items_for_training(dataset)
            self._generate_semantic_id_opq(sent_embs, sem_ids_path,
                                           training_item_mask)

        self.log(f'[TOKENIZER] Loading semantic IDs from {sem_ids_path}...')
        item2sem_ids = json.load(open(sem_ids_path, 'r'))
        item2sem_ids = {
            k: [int(x) for x in v]
            for k, v in item2sem_ids.items()
        }
        uniq = len({tuple(v) for v in item2sem_ids.values()})
        tot = len(item2sem_ids)
        rate = (tot - uniq) / tot if tot else 0.0
        self.log(f'[TOKENIZER] Loaded SIDs: {uniq}/{tot} unique, '
                 f'collision_rate={rate:.6f}')
        item2tokens = self._sem_ids_to_tokens(item2sem_ids)

        model_basename = os.path.basename(self.config['sent_emb_model'])
        map_tag = (f'{model_basename}_pca{SENT_EMB_PCA}_'
                   f'{self._quant_tag()}_{self.n_digit}d')
        self.item2tokens = item2tokens
        self.tokens2item = self._create_reverse_mapping()
        self._save_mappings(cache_dir, map_tag)
        return item2tokens

    def _create_reverse_mapping(self):
        tokens2item = {}
        for item, tokens in self.item2tokens.items():
            item_id = self.dataset.item2id[item]
            tokens2item[tuple(tokens)] = item_id
        return tokens2item

    def _save_mappings(self, cache_dir: str, map_tag: str):
        item_id2tokens = np.zeros((self.dataset.n_items, self.n_digit),
                                  dtype=np.int64)
        for item, tokens in self.item2tokens.items():
            item_id = self.dataset.item2id[item]
            item_id2tokens[item_id] = np.array(tokens)
        np.save(os.path.join(cache_dir, f'item_id2tokens_{map_tag}.npy'),
                item_id2tokens)
        with open(os.path.join(cache_dir, f'tokens2item_{map_tag}.pkl'),
                  'wb') as f:
            pickle.dump(self.tokens2item, f)
        self.log(f'[TOKENIZER] Saved mappings with tag: {map_tag}')
