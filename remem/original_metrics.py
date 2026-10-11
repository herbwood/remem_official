"""Original Stage 1 scoring and aggregation (eval_utils.py)."""
import logging
import re
from collections import defaultdict
import numpy as np
from rouge_score import rouge_scorer
logger = logging.getLogger(__name__)

def normalize_text(text: str) -> str:
    """Normalizes text for easier comparison."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub('[^a-z0-9\\s]', '', text)
    words = [word.rstrip('s') for word in text.split()]
    return ' '.join(words)

def calculate_and_summarize_metrics(results):
    """
    Receives a list of inference results, calculates ROUGE, Keyword Match Score, 
    AND Perplexity, then returns a summary.
    """
    logger.info('\n🧮 Calculating metrics...')
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    category_metrics = defaultdict(lambda : {'total': 0, 'correct_score': 0.0, 'rouge1': 0, 'rouge2': 0, 'rougeL': 0, 'perplexity_sum': 0.0, 'perplexity_count': 0})
    for res in results:
        ground_truth = res.get('ground_truth')
        if ground_truth is None:
            ground_truth = ''
        else:
            ground_truth = str(ground_truth)
        prediction = res.get('prediction')
        if prediction is None:
            prediction = ''
        else:
            prediction = str(prediction)
        rouge_scores = scorer.score(ground_truth, prediction)
        res['rouge1_f1'] = rouge_scores['rouge1'].fmeasure
        res['rouge2_f1'] = rouge_scores['rouge2'].fmeasure
        res['rougeL_f1'] = rouge_scores['rougeL'].fmeasure
        normalized_prediction = normalize_text(res['prediction'])
        raw_keywords = res.get('keywords', [])
        if isinstance(raw_keywords, str):
            keyword_list = [raw_keywords] if raw_keywords.strip() else []
        elif isinstance(raw_keywords, list):
            keyword_list = raw_keywords
        else:
            keyword_list = []
        if not keyword_list:
            match_score = 0.0
        else:
            matched_count = 0
            for k in keyword_list:
                normalized_k = normalize_text(k)
                if normalized_k and normalized_k in normalized_prediction:
                    matched_count += 1
            match_score = matched_count / len(keyword_list)
        res['keyword_match_score'] = match_score
        ppl = res.get('perplexity')
        has_valid_ppl = False
        if ppl is not None and (not np.isnan(ppl)):
            has_valid_ppl = True
        group_key = f"{res['question_type']}_{res['qa_category']}"
        category_metrics[group_key]['total'] += 1
        category_metrics[group_key]['correct_score'] += match_score
        category_metrics[group_key]['rouge1'] += res['rouge1_f1']
        category_metrics[group_key]['rouge2'] += res['rouge2_f1']
        category_metrics[group_key]['rougeL'] += res['rougeL_f1']
        if has_valid_ppl:
            category_metrics[group_key]['perplexity_sum'] += ppl
            category_metrics[group_key]['perplexity_count'] += 1
    summary = {'categories': {}}
    (total_score_sum, total_samples) = (0.0, 0)
    (total_r1, total_r2, total_rL) = (0.0, 0.0, 0.0)
    (total_ppl_sum, total_ppl_count) = (0.0, 0)
    for (group, metrics) in sorted(category_metrics.items()):
        total = metrics['total']
        avg_keyword_score = metrics['correct_score'] / total * 100 if total > 0 else 0
        ppl_count = metrics['perplexity_count']
        avg_ppl = metrics['perplexity_sum'] / ppl_count if ppl_count > 0 else float('nan')
        summary['categories'][group] = {'keyword_score': avg_keyword_score, 'avg_rouge1': metrics['rouge1'] / total if total > 0 else 0, 'avg_rouge2': metrics['rouge2'] / total if total > 0 else 0, 'avg_rougeL': metrics['rougeL'] / total if total > 0 else 0, 'avg_perplexity': avg_ppl, 'count': total, 'total_score': metrics['correct_score']}
        total_score_sum += metrics['correct_score']
        total_samples += total
        total_r1 += metrics['rouge1']
        total_r2 += metrics['rouge2']
        total_rL += metrics['rougeL']
        total_ppl_sum += metrics['perplexity_sum']
        total_ppl_count += metrics['perplexity_count']
    summary['overall'] = {'keyword_score': total_score_sum / total_samples * 100 if total_samples > 0 else 0, 'avg_rouge1': total_r1 / total_samples if total_samples > 0 else 0, 'avg_rouge2': total_r2 / total_samples if total_samples > 0 else 0, 'avg_rougeL': total_rL / total_samples if total_samples > 0 else 0, 'avg_perplexity': total_ppl_sum / total_ppl_count if total_ppl_count > 0 else float('nan'), 'total_samples': total_samples, 'total_score_sum': total_score_sum}
    return (results, summary)

def generate_keywords_list(attr, value_str):
    """
    Generates a list of keywords based on specific parsing rules per attribute.
    value_str is the formatted string (e.g., "29,500" or "Has a 401(k)").
    """
    if attr in ['date_of_birth', 'phone_number']:
        return value_str.split('-')
    if attr in ['annual_salary_usd', 'email', 'gender', 'marital_status', 'race_or_ethnicity']:
        return [value_str]
    if attr == 'financial_tag':
        stopwords = {'a', 'in'}
        words = value_str.split()
        return [w for w in words if w.lower() not in stopwords]
    return value_str.split()
