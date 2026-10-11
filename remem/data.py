"""One Hugging Face interface for ReMem; image_path is metadata only."""
import argparse
import ast
import hashlib
import itertools
import json
import random
import re
from collections import Counter
from pathlib import Path

DATASET_ID = 'herbwood27/Remem'
DEFAULT_REVISION = '60d778bf1ebb612ff22fd4c7673e6b65563d1c89'
SPLITS = {'finetune': 2000, 'forget1': 100, 'forget2': 200,
          'forget3': 300, 'forget4': 400, 'retain': 560, 'test': 560}
FIELDS = ('image', 'question', 'answer', 'keywords', 'question_type',
          'qa_category', 'attribute', 'cloze_prompt', 'image_path')
CONFIG_DIR = Path(__file__).resolve().parents[1] / 'configs'

def forget_split(ratio):
    if ratio not in (5, 10, 15, 20):
        raise ValueError('Forget ratio must be 5, 10, 15, or 20 (percent).')
    return f'forget{ratio // 5}'

def parse_keywords(raw):
    """HF string is either a scalar value or a serialized Python list."""
    if isinstance(raw, str):
        if not raw.strip():
            raise ValueError('Empty keywords')
        if raw.lstrip().startswith('['):
            try:
                raw = ast.literal_eval(raw)
            except (ValueError, SyntaxError) as exc:
                raise ValueError('Invalid serialized keyword list') from exc
        else:
            raw = [raw]
    if not isinstance(raw, list) or not raw or any(
            not isinstance(k, str) or not k.strip() for k in raw):
        raise ValueError('Keywords must contain nonempty strings')
    return list(raw)

def identity(row):
    """Audited original train/test filename convention, never an image source."""
    path = row['image_path']
    if not isinstance(path, str) or '/' in path or '\\' in path:
        raise ValueError('Expected a published image filename')
    match = re.fullmatch(r'(.+?)(?:_test)?_\d+\.png', path)
    if not match:
        raise ValueError(f'Unverified identity filename convention: {path!r}')
    return match.group(1)

def prompt(question):
    return f'USER: <image>\n{question}\nASSISTANT:'

def validate_record(sample, *, require_image=True):
    missing = set(FIELDS) - set(sample)
    if missing:
        raise ValueError(f'Missing ReMem columns: {sorted(missing)}')
    for key in FIELDS[1:]:
        if key != 'keywords' and (not isinstance(sample[key], str) or not sample[key].strip()):
            raise ValueError(f'Invalid or empty {key}')
    if sample['question_type'] not in ('1-hop', 'multi-hop'):
        raise ValueError(f'Unknown question_type: {sample["question_type"]}')
    if sample['qa_category'] not in ('identity', 'personal_information'):
        raise ValueError('Unknown qa_category')
    keywords = parse_keywords(sample['keywords'])
    identity(sample)
    if require_image:
        from PIL import Image
        if not isinstance(sample['image'], Image.Image):
            raise ValueError('sample["image"] must be a decoded PIL image')
        sample['image'].load()
        if min(sample['image'].size) <= 0:
            raise ValueError('Invalid decoded image dimensions')
    result = dict(sample)
    result['keywords_raw'] = sample['keywords']
    result['keywords'] = keywords
    if require_image:
        result['image'] = sample['image'].convert('RGB')
    return result

def load_split(split, *, revision=DEFAULT_REVISION, cache_dir=None, streaming=False):
    if split not in SPLITS:
        raise ValueError(f'Unsupported split: {split}')
    from datasets import load_dataset
    ds = load_dataset(DATASET_ID, split=split, revision=revision,
                      cache_dir=cache_dir, streaming=streaming)
    if ds.features is None or set(FIELDS) - set(ds.features):
        raise ValueError('Dataset features do not match ReMem')
    from datasets import Image, Value
    if not isinstance(ds.features['image'], Image):
        raise ValueError('Expected HF Image feature')
    for key in FIELDS[1:]:
        if not isinstance(ds.features[key], Value) or ds.features[key].dtype != 'string':
            raise ValueError(f'Expected published string feature for {key}')
    if not streaming and len(ds) != SPLITS[split]:
        raise ValueError(f'Unexpected row count for {split}: {len(ds)}')
    return ds

def metadata_digest(rows):
    # Excludes encoded image bytes and the original-only canonical keyword column.
    records = [{key: row[key] for key in FIELDS[1:]} for row in rows]
    for row in records:
        row['keywords'] = parse_keywords(row['keywords'])
    serialized = [json.dumps(row, sort_keys=True, ensure_ascii=False) for row in records]
    return hashlib.sha256('\n'.join(sorted(serialized)).encode()).hexdigest()

def validate_metadata_membership(rows, split):
    """Compare complete text metadata with the audited source manifest."""
    manifest = json.loads((CONFIG_DIR / 'protocol.json').read_text(encoding='utf-8'))
    expected = manifest['splits'][split]
    if len(rows) != expected['count'] or metadata_digest(rows) != expected['metadata_sha256']:
        raise ValueError(f'{split} membership differs from audited protocol; re-audit this revision.')
    for row in rows:
        validate_record({**row, 'image': None}, require_image=False)

def metadata_rows(ds):
    from datasets import Image
    return [dict(row) for row in ds.cast_column('image', Image(decode=False)).remove_columns('image')]

def get_forget_identities(ratio):
    names = json.loads((CONFIG_DIR / 'forget_identities.json').read_text(encoding='utf-8'))
    forget_split(ratio)
    return set(names[str(ratio // 5)])

def stage2_partition(finetune_rows, published_forget_rows, ratio):
    """Original identity filter on full training QAs, with verified HF membership."""
    split = forget_split(ratio)
    validate_metadata_membership(finetune_rows, 'finetune')
    validate_metadata_membership(published_forget_rows, split)
    names = get_forget_identities(ratio)
    actual = Counter(identity(row) for row in published_forget_rows)
    if set(actual) != names or any(n != 100 for n in actual.values()):
        raise ValueError('Published forget identities do not match source identity manifests')
    forget = [i for i, row in enumerate(finetune_rows) if identity(row) in names]
    retain = [i for i, row in enumerate(finetune_rows) if identity(row) not in names]
    if len(forget) != SPLITS[split] or len(retain) != 2000 - SPLITS[split]:
        raise ValueError('Unexpected identity training membership')
    return forget, retain

def process_item(processor, row, answer=None):
    """Original data_module.py answer-only label masking and instruction format."""
    row = validate_record(row)
    prompt_text = prompt(row['question'])
    full_text = f'{prompt_text} {row["answer"] if answer is None else answer}'
    inputs = processor(text=full_text, images=row['image'], return_tensors='pt')
    inputs = {k: v.squeeze(0) for k, v in inputs.items()}
    prompt_inputs = processor(text=prompt_text, images=row['image'], return_tensors='pt')
    prompt_len = len(prompt_inputs['input_ids'].squeeze(0))
    labels = inputs['input_ids'].clone()
    labels[:prompt_len] = -100
    inputs['labels'] = labels
    return inputs

class MemorizationDataset:
    def __init__(self, rows, processor):
        self.rows, self.processor = rows, processor
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        return process_item(self.processor, self.rows[idx])

class UnlearningDataset:
    def __init__(self, rows, forget_indices, retain_indices, processor, method, idk_responses=None):
        self.rows, self.forget, self.retain = rows, forget_indices, retain_indices
        self.processor, self.method = processor, method
        self.idk_responses = idk_responses
        if method in ('DPO', 'NPO') and (not idk_responses or any(
                not isinstance(x, str) or not x.strip() for x in idk_responses)):
            raise ValueError('DPO/NPO require the original IDK response list; none is bundled.')
        if method in ('GD', 'KL') and not retain_indices:
            raise ValueError('Retain training subset is empty')
    def __len__(self):
        return len(self.forget)
    def __getitem__(self, idx):
        row = self.rows[self.forget[idx]]
        if self.method in ('DPO', 'NPO'):
            # Original code samples IDK for both methods, although NPO ignores it.
            idk = random.choice(self.idk_responses)
            return {'idk_sample': process_item(self.processor, row, idk),
                    'forget_sample': process_item(self.processor, row)}
        result = {'forget_sample': process_item(self.processor, row)}
        if self.method != 'GA':
            rand_idx = (idx + random.randint(0, len(self.retain) - 1)) % len(self.retain)
            result['retain_sample'] = process_item(self.processor, self.rows[self.retain[rand_idx]])
        return result

def main():
    parser = argparse.ArgumentParser(description='Inspect small streamed ReMem samples; no full download.')
    parser.add_argument('--split', choices=list(SPLITS), default='finetune')
    parser.add_argument('--revision', default=DEFAULT_REVISION)
    parser.add_argument('--cache-dir')
    parser.add_argument('--limit', type=int, default=1)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit must be positive')
    ds = load_split(args.split, revision=args.revision, cache_dir=args.cache_dir, streaming=True)
    print(ds.features)
    for sample in itertools.islice(ds, args.limit):
        row = validate_record(sample)
        row['image'] = {'size': row['image'].size, 'mode': row['image'].mode}
        print(json.dumps(row, ensure_ascii=False))

if __name__ == '__main__':
    main()
