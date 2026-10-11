"""Explicit metric variants and validated Exposure targets."""
import json
import re
from functools import lru_cache
from .data import CONFIG_DIR, identity, parse_keywords

def normalize_stage2(text):
    # Original eval_unlearn.py normalize_text, including its plural stripping.
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return ' '.join(word.rstrip('s') for word in text.split())

def get_hop_category(q_type):
    q_type = str(q_type).lower()
    if '1-hop' in q_type or 'single' in q_type:
        return 'single-hop'
    if 'multi' in q_type or '2-hop' in q_type or '3-hop' in q_type:
        return 'multi-hop'
    return 'unknown'

def stage1_metrics(results):
    from .original_metrics import calculate_and_summarize_metrics
    return calculate_and_summarize_metrics(results)

def stage2_em(raw_keywords, prediction):
    """Original normalized phrase-containment score, not whole-answer equality.

    In the original finetune/retain/test files a list was coerced to a string;
    HF serializes exactly that list. Preserve that behavior, including the
    distinction from original scalar keywords in published forget splits.
    """
    return normalize_stage2(raw_keywords) in normalize_stage2(prediction)

def all_keywords_match(keywords, prediction):
    """Diagnostic only; deliberately not substituted for the source EM."""
    normalized = normalize_stage2(prediction)
    return all(normalize_stage2(k) and normalize_stage2(k) in normalized for k in keywords)

@lru_cache(maxsize=1)
def exposure_metadata():
    candidates = json.loads((CONFIG_DIR / 'attribute_to_values.json').read_text(encoding='utf-8'))
    targets = json.loads((CONFIG_DIR / 'exposure_targets.json').read_text(encoding='utf-8'))
    for attr, values in candidates.items():
        if not values or len(values) != len(set(values)) or any(not isinstance(c, str) or not c.strip() for c in values):
            raise ValueError(f'Invalid original candidate set: {attr}')
        if len({c.casefold() for c in values}) != len(values):
            raise ValueError(f'Ambiguous case-normalized candidates: {attr}')
    return candidates, targets

def resolve_exposure_target(row):
    from .original_metrics import generate_keywords_list
    candidates, targets = exposure_metadata()
    attr = row['attribute']
    key = identity(row) + '/' + attr
    if attr not in candidates or key not in targets:
        raise ValueError(f'No verified original Exposure metadata for {key}')
    target = targets[key]
    parsed = parse_keywords(row.get('keywords_raw', row['keywords']))
    expected = generate_keywords_list(attr, target)
    # Earlier forget QAs use a scalar canonical value. Current QAs use parsed lists.
    if [k.casefold() for k in parsed] != [k.casefold() for k in expected] and not (len(parsed) == 1 and parsed[0].casefold() == target.casefold()):
        raise ValueError(f'Published keywords disagree with original target: {key}')
    if normalize_stage2(target) not in normalize_stage2(row['answer']):
        raise ValueError(f'Published answer disagrees with original target: {key}')
    matches = [value for value in candidates[attr] if value.casefold() == target.casefold()]
    if len(matches) != 1 or not row['cloze_prompt'].strip():
        raise ValueError(f'Ambiguous or missing Exposure target: {key}')
    return matches[0], candidates[attr]

def aggregate_stage2(records):
    summary = {}
    for metric in ('retain_em', 'retain_rouge', 'forget_em', 'test_em', 'exposure'):
        role = metric.split('_')[0]
        field = 'rougeL' if metric == 'retain_rouge' else 'exposure' if metric == 'exposure' else 'em'
        rows = records.get(role, [])
        summary[metric] = {}
        for label, hop in [('overall', None), ('single_hop', 'single-hop'), ('multi_hop', 'multi-hop')]:
            values = [row[field] for row in rows if field in row and (hop is None or row['hop'] == hop)]
            summary[metric][label] = sum(values) / len(values) if values else None
        summary[metric]['count'] = sum(field in row for row in rows)
    return summary
