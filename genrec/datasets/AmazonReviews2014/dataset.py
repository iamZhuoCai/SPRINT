import ast
import gzip
import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List

from tqdm import tqdm

from genrec.dataset import AbstractDataset
from genrec.utils import clean_text, download_file


class AmazonReviews2014(AbstractDataset):
    """Amazon Reviews 2014 datasets used by TIGER / DiffGRM.

    Raw sources (Stanford SNAP / UCSD McAuley), SNAP's prefiltered 5-core
    review dumps plus item metadata:
      - Beauty:  reviews_Beauty_5.json.gz, meta_Beauty.json.gz
      - Toys:    reviews_Toys_and_Games_5.json.gz, meta_Toys_and_Games.json.gz
      - Sports:  reviews_Sports_and_Outdoors_5.json.gz,
                 meta_Sports_and_Outdoors.json.gz

    Processing follows TIGER's leave-one-out protocol:
      sort each user's interactions by timestamp, then
      train=[:-2], val=[:-1], test=full sequence.
    """

    SNAP_BASE_URL = (
        'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles')

    CATEGORY_FILES = {
        'Beauty': {
            'reviews': 'reviews_Beauty_5.json.gz',
            'meta': 'meta_Beauty.json.gz',
        },
        'Toys': {
            'reviews': 'reviews_Toys_and_Games_5.json.gz',
            'meta': 'meta_Toys_and_Games.json.gz',
        },
        'Sports': {
            'reviews': 'reviews_Sports_and_Outdoors_5.json.gz',
            'meta': 'meta_Sports_and_Outdoors.json.gz',
        },
    }

    def __init__(self, config: dict):
        super(AmazonReviews2014, self).__init__(config)

        self.category = config['category']
        self._check_available_category()
        self.log(
            f'[DATASET] Amazon Reviews 2014 for category: {self.category} '
            f'(5-core)')

        self.cache_dir = os.path.join(config['cache_dir'], 'AmazonReviews2014',
                                      self.category)
        self._download_and_process_raw()

    def _check_available_category(self):
        available = list(self.CATEGORY_FILES.keys())
        assert self.category in available, (
            f'Category "{self.category}" not available. '
            f'Available categories: {available}')

    def _raw_paths(self) -> Dict[str, str]:
        files = self.CATEGORY_FILES[self.category]
        raw_dir = os.path.join(self.cache_dir, 'raw')
        return {
            'raw_dir': raw_dir,
            'reviews': os.path.join(raw_dir, files['reviews']),
            'meta': os.path.join(raw_dir, files['meta']),
        }

    def _download_large_file(self, url: str, path: str):
        """Stream large Amazon 2014 dumps to disk."""
        import requests

        self.log(f'[DATASET] Downloading {url}')
        with requests.get(url, stream=True, timeout=600) as response:
            if response.status_code != 200:
                raise FileNotFoundError(
                    f'Failed to download {url} '
                    f'(status={response.status_code}). '
                    f'Please manually download and place it at {path}')
            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        self.log(f'[DATASET] Downloaded {os.path.basename(path)}')

    def _ensure_raw_files(self):
        paths = self._raw_paths()
        os.makedirs(paths['raw_dir'], exist_ok=True)
        needed = {
            'reviews': os.path.basename(paths['reviews']),
            'meta': os.path.basename(paths['meta']),
        }
        for key, filename in needed.items():
            local_path = paths[key]
            if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                continue
            url = f"{self.SNAP_BASE_URL}/{filename}"
            tmp_path = local_path + '.partial'
            try:
                self._download_large_file(url, tmp_path)
                os.replace(tmp_path, local_path)
            except Exception:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                # Fallback to non-streaming helper.
                download_file(url, local_path)
            if not os.path.exists(local_path) or os.path.getsize(
                    local_path) == 0:
                raise FileNotFoundError(
                    f'Failed to download {filename}. '
                    f'Please manually download from {url} and place it at '
                    f'{local_path}')

    @staticmethod
    def _parse_line(line: str) -> dict:
        line = line.strip()
        if not line:
            return {}
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            # Amazon 2014 metadata often uses Python literals.
            return ast.literal_eval(line)

    def _iter_gzip_json(self, path: str) -> Iterable[dict]:
        with gzip.open(path, 'rt', encoding='utf-8', errors='ignore') as fp:
            for line in fp:
                obj = self._parse_line(line)
                if obj:
                    yield obj

    def _load_interactions(self, reviews_path: str):
        user_item_ts = defaultdict(list)
        for review in tqdm(self._iter_gzip_json(reviews_path),
                           desc='Reading reviews'):
            user = review.get('reviewerID')
            item = review.get('asin')
            ts = review.get('unixReviewTime')
            if user is None or item is None or ts is None:
                continue
            user_item_ts[user].append((item, int(ts)))

        all_item_seqs = {}
        for user, interactions in user_item_ts.items():
            interactions.sort(key=lambda x: (x[1], x[0]))
            # Keep chronological order; drop exact consecutive duplicates.
            seq = []
            for item, _ in interactions:
                if not seq or seq[-1] != item:
                    seq.append(item)
            if len(seq) >= 3:
                all_item_seqs[user] = seq
        return all_item_seqs

    def _build_id_mapping(self, all_item_seqs: Dict[str, List[str]],
                          output_path: str) -> dict:
        id_mapping_file = os.path.join(output_path, 'id_mapping.json')
        if os.path.exists(id_mapping_file):
            self.log(f'[DATASET] Loading id mapping from {id_mapping_file}')
            return json.load(open(id_mapping_file, 'r'))

        self.log('[DATASET] Creating id mapping')
        id_mapping = {
            'user2id': {
                '[PAD]': 0
            },
            'item2id': {
                '[PAD]': 0
            },
            'id2user': ['[PAD]'],
            'id2item': ['[PAD]'],
        }
        for user, items in all_item_seqs.items():
            if user not in id_mapping['user2id']:
                id_mapping['user2id'][user] = len(id_mapping['user2id'])
                id_mapping['id2user'].append(user)
            for item in items:
                if item not in id_mapping['item2id']:
                    id_mapping['item2id'][item] = len(id_mapping['item2id'])
                    id_mapping['id2item'].append(item)

        with open(id_mapping_file, 'w') as f:
            json.dump(id_mapping, f)
        return id_mapping

    def _feature_process(self, feature) -> str:
        if feature is None:
            return ''
        if isinstance(feature, float) or isinstance(feature, int):
            return f'{feature}. '
        if isinstance(feature, dict):
            # e.g. salesRank: {"Beauty": 123}
            parts = []
            for k, v in feature.items():
                parts.append(f'{k}: {v}')
            text = clean_text(', '.join(parts))
            return f'{text}. ' if text else ''
        if isinstance(feature, list):
            if not feature:
                return ''
            # nested category lists: [["Beauty", "Hair"], ...]
            flat = []
            for v in feature:
                if isinstance(v, list):
                    flat.append(' > '.join(map(str, v)))
                else:
                    flat.append(str(v))
            text = clean_text(', '.join(flat))
            return f'{text}. ' if text else ''
        text = clean_text(feature)
        return f'{text}. ' if text else ''

    def _meta_to_sentence(self, metadata: dict) -> str:
        # Align with TIGER process.ipynb semantic fields, plus description.
        features_needed = [
            'title', 'price', 'salesRank', 'brand', 'categories', 'description'
        ]
        sentence = ''
        for feature in features_needed:
            sentence += self._feature_process(metadata.get(feature))
        return sentence.strip()

    def _process_meta(self, meta_path: str, output_path: str,
                      item2id: Dict[str, int]) -> dict:
        meta_file = os.path.join(output_path, 'metadata.sentence.json')
        if os.path.exists(meta_file):
            self.log(f'[DATASET] Metadata has been processed...')
            return json.load(open(meta_file, 'r'))

        self.log('[DATASET] Processing metadata, mode: sentence')
        item2meta = {}
        keep_asins = set(item2id.keys()) - {'[PAD]'}
        for metadata in tqdm(self._iter_gzip_json(meta_path),
                             desc='Reading metadata'):
            asin = metadata.get('asin')
            if asin not in keep_asins:
                continue
            item2meta[asin] = self._meta_to_sentence(metadata)

        self.log(
            f'[DATASET] {len(item2meta)} of {len(keep_asins)} items have meta data.'
        )
        # Fill missing metadata with empty string so tokenizer can still encode.
        for asin in keep_asins:
            if asin not in item2meta:
                item2meta[asin] = ''

        with open(meta_file, 'w') as f:
            json.dump(item2meta, f)
        return item2meta

    @staticmethod
    def _processed_file_ready(path: str) -> bool:
        return os.path.isfile(path) and os.path.getsize(path) > 0

    def _download_and_process_raw(self):
        """Load each processed artifact if it is already on disk.

        A missing or empty file is rebuilt from the raw SNAP dumps. Rebuilding
        interaction sequences also rebuilds the id mapping and metadata,
        because those two are derived from the sequence file.
        """
        processed_data_path = os.path.join(self.cache_dir, 'processed')
        os.makedirs(processed_data_path, exist_ok=True)
        seq_file = os.path.join(processed_data_path, 'all_item_seqs.json')
        id_mapping_file = os.path.join(processed_data_path, 'id_mapping.json')
        meta_file = os.path.join(processed_data_path, 'metadata.sentence.json')

        need_seq = not self._processed_file_ready(seq_file)
        need_ids = need_seq or not self._processed_file_ready(id_mapping_file)
        need_meta = need_ids or not self._processed_file_ready(meta_file)
        paths = None
        if need_seq or need_meta:
            self._ensure_raw_files()
            paths = self._raw_paths()

        if need_seq:
            self.log('[DATASET] all_item_seqs.json missing; '
                     'rebuilding from raw reviews')
            self.all_item_seqs = self._load_interactions(paths['reviews'])
            with open(seq_file, 'w') as f:
                json.dump(self.all_item_seqs, f)
        else:
            self.log(f'[DATASET] Loading all_item_seqs from {seq_file}')
            self.all_item_seqs = json.load(open(seq_file, 'r'))

        if need_ids:
            if os.path.exists(id_mapping_file):
                os.remove(id_mapping_file)
            self.log('[DATASET] id_mapping.json missing; rebuilding')
            self.id_mapping = self._build_id_mapping(self.all_item_seqs,
                                                     processed_data_path)
        else:
            self.log(f'[DATASET] Loading id mapping from {id_mapping_file}')
            self.id_mapping = json.load(open(id_mapping_file, 'r'))

        if need_meta:
            if os.path.exists(meta_file):
                os.remove(meta_file)
            self.log(f'[DATASET] {os.path.basename(meta_file)} missing; '
                     'reprocessing metadata')
            self.item2meta = self._process_meta(paths['meta'],
                                                processed_data_path,
                                                self.id_mapping['item2id'])
        else:
            self.log(f'[DATASET] Loading metadata from {meta_file}')
            self.item2meta = json.load(open(meta_file, 'r'))
