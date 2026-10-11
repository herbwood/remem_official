"""Validate all text metadata without fetching 9.53 GB of image shards."""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from remem.data import (DATASET_ID, DEFAULT_REVISION, FIELDS, SPLITS, identity,
                        load_split, stage2_partition, validate_metadata_membership,
                        validate_record)
from remem.utils import save_json

def get_json(url, params=None):
    import requests
    for attempt in range(5):
        response = requests.get(url, params=params, timeout=60)
        if response.status_code != 429:
            response.raise_for_status()
            return response.json()
        if attempt == 4:
            response.raise_for_status()
        delay = min(60, max(15 * (attempt + 1), int(response.headers.get('Retry-After', '15'))))
        print(f'Dataset viewer rate limit; retrying in {delay}s.', flush=True)
        time.sleep(delay)

def fetch_metadata(split):
    rows = []
    for offset in range(0, SPLITS[split], 100):
        cached = Path('.cache') / 'viewer' / DEFAULT_REVISION / f'{split}-{offset}.json'
        if cached.is_file():
            result = json.loads(cached.read_text(encoding='utf-8'))
        else:
            result = get_json('https://datasets-server.huggingface.co/rows',
                          {'dataset': DATASET_ID, 'config': 'default', 'split': split,
                           'offset': offset, 'length': min(100, SPLITS[split] - offset)})
            # Cache text only; omit signed asset URLs.
            for item in result['rows']:
                item['row'].pop('image', None)
            save_json(cached, result)
            time.sleep(2)
        if result.get('partial') or result['num_rows_total'] != SPLITS[split]:
            raise ValueError('Incomplete dataset viewer response')
        for item in result['rows']:
            if item['truncated_cells']:
                raise ValueError('Truncated dataset metadata')
            rows.append({k: v for k, v in item['row'].items() if k != 'image'})
    validate_metadata_membership(rows, split)
    return split, rows

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='validation/remote.json')
    parser.add_argument('--cache-dir')
    parser.add_argument('--metadata-only', action='store_true')
    args = parser.parse_args()
    repo = get_json(f'https://huggingface.co/api/datasets/{DATASET_ID}')
    # The viewer only exposes HEAD. Refuse to use it for a different revision.
    if repo['sha'] != DEFAULT_REVISION:
        raise ValueError('HF HEAD changed; re-audit source membership for the new revision.')
    info = get_json('https://datasets-server.huggingface.co/info', {'dataset': DATASET_ID})
    features = info['dataset_info']['default']['features']
    expected_features = {k: {'dtype': 'string', '_type': 'Value'} for k in FIELDS[1:]}
    expected_features['image'] = {'_type': 'Image'}
    assert features == expected_features
    counts = {k: v['num_examples'] for k, v in info['dataset_info']['default']['splits'].items()}
    assert counts == SPLITS
    metadata = {}
    for split in SPLITS:
        key, rows = fetch_metadata(split)
        metadata[key] = rows
        print(f'Validated {split}: {len(rows)} text records.', flush=True)
    assert get_json(f'https://huggingface.co/api/datasets/{DATASET_ID}')['sha'] == repo['sha']
    partitions = {}
    for ratio in (5, 10, 15, 20):
        forget, retain = stage2_partition(metadata['finetune'], metadata[f'forget{ratio // 5}'], ratio)
        partitions[str(ratio)] = {'forget_train': len(forget), 'retain_train': len(retain)}
    from remem.metrics import resolve_exposure_target
    target_checks = {}
    for split, rows in metadata.items():
        failures = []
        for row in rows:
            try:
                resolve_exposure_target(validate_record({**row, 'image': None}, require_image=False))
            except ValueError as exc:
                failures.append(str(exc))
        target_checks[split] = {'validated': len(rows) - len(failures), 'mismatches': len(failures),
                               'interpretation': 'strict canonical string checks; earlier forget salaries include a dollar prefix',
                               'examples': list(dict.fromkeys(failures))[:5]}
        if failures and split in ('finetune', 'retain', 'test'):
            raise ValueError(f'Canonical targets disagree with the active protocol in {split}: {failures[:1]}')
    report = {'revision': repo['sha'], 'features': features, 'counts': counts,
              'metadata_membership': 'all 4120 rows match original QA metadata',
              'identity_counts': {k: dict(Counter(identity(r) for r in v)) for k, v in metadata.items()},
              'partitions': partitions, 'exposure_target_checks': target_checks,
              'streaming_image': 'not run (--metadata-only)'}
    if not args.metadata_only:
        ds = load_split('finetune', revision=repo['sha'], streaming=True, cache_dir=args.cache_dir)
        row = validate_record(next(iter(ds)))
        report['streaming_image'] = {'size': row['image'].size, 'mode': row['image'].mode,
                                     'keywords': row['keywords'], 'question_type': row['question_type']}
    save_json(args.output, report)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
