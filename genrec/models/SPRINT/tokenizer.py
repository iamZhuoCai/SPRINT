import os
import shutil

import yaml
from datasets import DatasetDict, load_from_disk

from genrec.constants import MAX_ITEM_SEQ_LEN, MIN_HIST_LEN, SENT_EMB_PCA
from genrec.dataset import AbstractDataset
from genrec.tokenizer import AbstractTokenizer
from genrec.tokenizers.OPQ.tokenizer import OPQTokenizer


class SPRINTTokenizer(AbstractTokenizer):

    def __init__(self, config: dict, dataset: AbstractDataset):
        super(SPRINTTokenizer, self).__init__(config, dataset)
        self.accelerator = config['accelerator']
        self.dataset = dataset

        self.log(f'[TOKENIZER] Loading tokenizer config')
        tokenizer_config: dict = yaml.safe_load(
            open('genrec/tokenizers/OPQ/config.yaml', 'r'))

        for key in tokenizer_config.keys():
            if key in config.keys():
                tokenizer_config[key] = config[key]
            self.log(f"{key}: {tokenizer_config[key]}")
        config.update(tokenizer_config)
        self.config = config

        self.tokenizer = OPQTokenizer(config, dataset)

        self.item2id = dataset.item2id
        self.item2tokens = self.tokenizer.item2tokens
        # DiffGRM puts MASK after SID slices; MultiHeadVQVAE uses eos+1.
        if getattr(self.tokenizer, 'mask_token_id', None) is not None:
            self.mask_token = int(self.tokenizer.mask_token_id)
        else:
            self.mask_token = self.tokenizer.eos_token + 1
        self.ignored_label = self.tokenizer.ignored_label

    @property
    def n_digit(self):
        return self.tokenizer.n_digit

    @property
    def codebook_sizes(self):
        return self.tokenizer.codebook_sizes

    @property
    def max_token_seq_len(self) -> int:
        return MAX_ITEM_SEQ_LEN

    @property
    def vocab_size(self) -> int:
        """
        Returns the vocabulary size for the TIGER tokenizer.
        """
        return self.mask_token + 1

    def _tokenize_items(self, item_seq: list, test=False):
        input_ids = [self.item2id[item] for item in item_seq[:-1]]
        seq_lens = len(input_ids)
        attention_mask = [1] * seq_lens

        pad_lens = self.max_token_seq_len - seq_lens
        input_ids = [0] * pad_lens + input_ids
        attention_mask = [0] * pad_lens + attention_mask

        if test:
            labels = list(self.item2tokens[item_seq[-1]])
        else:
            labels = [self.item2id[item_seq[-1]]]

        return input_ids, attention_mask, labels, seq_lens

    def _tokenizer_right_slide(self) -> bool:
        """Whether legacy tokenizer right-slide expansion is enabled."""
        val = self.config.get('tokenizer_right_slide', True)
        return val if isinstance(val, bool) else str(val).lower() == 'true'

    def tokenize_function(self, example: dict, split: str) -> dict:
        """Single sample (batched=False), no window expansion."""
        max_item_seq_len = MAX_ITEM_SEQ_LEN
        item_seq = example['item_seq']
        cur_item_seq = item_seq[-(max_item_seq_len + 1):]
        input_ids, attention_mask, labels, seq_lens = self._tokenize_items(
            cur_item_seq, test=(split != 'train'))
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'seq_lens': seq_lens,
        }

    def tokenize_function_right_slide(self, example: dict, split: str) -> dict:
        """Legacy tokenizer right-aligned sliding (batched=True, batch_size=1).

        Expands leave-one-out train ``seq[:-2]`` into windows ending at each
        position i where history length is at least ``min_hist_len``, cropped
        to ``max_item_seq_len`` history items.
        """
        max_item_seq_len = MAX_ITEM_SEQ_LEN
        min_hist_len = MIN_HIST_LEN
        item_seq = example['item_seq'][0]

        if split == 'train':
            all_input_ids, all_attention_mask, all_labels, all_seq_lens = (
                [], [], [], [])
            start_i = max(2, min_hist_len + 1)
            for i in range(start_i, len(item_seq) + 1):
                cur_item_seq = item_seq[max(0, i - max_item_seq_len - 1):i]
                input_ids, attention_mask, labels, seq_lens = (
                    self._tokenize_items(cur_item_seq))
                all_input_ids.append(input_ids)
                all_attention_mask.append(attention_mask)
                all_labels.append(labels)
                all_seq_lens.append(seq_lens)
            return {
                'input_ids': all_input_ids,
                'attention_mask': all_attention_mask,
                'labels': all_labels,
                'seq_lens': all_seq_lens,
            }

        input_ids, attention_mask, labels, seq_lens = self._tokenize_items(
            item_seq[-(max_item_seq_len + 1):], test=True)
        return {
            'input_ids': [input_ids],
            'attention_mask': [attention_mask],
            'labels': [labels],
            'seq_lens': [seq_lens],
        }

    def _sid_map_tag(self) -> str:
        """Tag aligned with DiffGRM item_id2tokens_* so SID changes invalidate cache."""
        inner = self.tokenizer
        if hasattr(inner, '_quant_tag'):
            model_basename = os.path.basename(
                self.config.get('sent_emb_model', 'emb'))
            pca = SENT_EMB_PCA
            return (f'{model_basename}_pca{pca}_'
                    f'{inner._quant_tag()}_{self.n_digit}d')
        return f'OPQ_{self.n_digit}d'

    def _tokenized_cache_dir(self) -> str:
        tag = self._sid_map_tag().replace(',', '_').replace('/', '_')
        seq = MAX_ITEM_SEQ_LEN
        if self._tokenizer_right_slide():
            # Leave-one-out + tokenizer right-aligned windows.
            slide_tag = f'_tokrightslide_h{MIN_HIST_LEN}'
        else:
            # Leave-one-out with one sample per example (no tokenizer slide).
            slide_tag = '_toknoslide'
        return os.path.join(self.dataset.cache_dir, 'processed',
                            f'tokenized_seq{seq}_{tag}{slide_tag}')

    @staticmethod
    def _tokenized_cache_ready(cache_dir: str, splits) -> bool:
        if not os.path.exists(os.path.join(cache_dir, 'dataset_dict.json')):
            return False
        return all(
            os.path.isdir(os.path.join(cache_dir, split)) for split in splits)

    def tokenize(self, datasets: dict) -> dict:
        cache_dir = self._tokenized_cache_dir()
        force = bool(self.config.get('force_retokenize', False))
        splits = list(datasets.keys())
        tok_right_slide = self._tokenizer_right_slide()

        def _build_and_maybe_save():
            is_main = (self.accelerator is None
                       or self.accelerator.is_main_process)

            if is_main and force and os.path.isdir(cache_dir):
                self.log(
                    f'[TOKENIZER] force_retokenize=True, removing {cache_dir}')
                shutil.rmtree(cache_dir)

            if self._tokenized_cache_ready(cache_dir, splits):
                self.log(
                    f'[TOKENIZER] Loading tokenized datasets from {cache_dir}')
                loaded = load_from_disk(cache_dir)
                return {split: loaded[split] for split in splits}

            if tok_right_slide:
                self.log(
                    '[TOKENIZER] leave-one-out + tokenizer right-aligned '
                    'sliding windows')
            else:
                self.log(
                    '[TOKENIZER] tokenizer_right_slide=False: leave-one-out '
                    'with no tokenizer sliding expansion')

            # datasets>=2 still forks a worker for num_proc=1; on large
            # corpora that fork (parent RSS in the tens of GB) gets the child
            # killed mid-map (EOFError from iflatmap_unordered). Pass None so
            # the map runs in-process.
            map_num_proc = self.config['num_proc']
            if map_num_proc is not None and int(map_num_proc) <= 1:
                map_num_proc = None

            tokenized_datasets = {}
            for split in splits:
                if tok_right_slide:
                    # Legacy: expand train via batched map returning N windows.
                    tokenized_datasets[split] = datasets[split].map(
                        lambda t, s=split: self.tokenize_function_right_slide(
                            t, s),
                        batched=True,
                        batch_size=1,
                        remove_columns=datasets[split].column_names,
                        num_proc=map_num_proc,
                        desc=f'Tokenizing {split} set: ')
                else:
                    tokenized_datasets[split] = datasets[split].map(
                        lambda t, s=split: self.tokenize_function(t, s),
                        batched=False,
                        remove_columns=datasets[split].column_names,
                        num_proc=map_num_proc,
                        desc=f'Tokenizing {split} set: ')

            if is_main:
                os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
                if os.path.isdir(cache_dir):
                    shutil.rmtree(cache_dir)
                self.log(
                    f'[TOKENIZER] Saving tokenized datasets to {cache_dir}')
                DatasetDict(tokenized_datasets).save_to_disk(cache_dir)

            return tokenized_datasets

        if self.accelerator is not None:
            with self.accelerator.main_process_first():
                tokenized_datasets = _build_and_maybe_save()
        else:
            tokenized_datasets = _build_and_maybe_save()

        for split in splits:
            tokenized_datasets[split].set_format(type='torch')
            self.log(
                f'Tokenized {split} set: {len(tokenized_datasets[split])}')

        return tokenized_datasets
