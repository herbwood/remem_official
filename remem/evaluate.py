"""Original Stage 1/2 inference, metric variants, and identity evaluation roles."""
import argparse
import itertools
from .data import (SPLITS, forget_split, get_forget_identities, identity, load_split,
                   metadata_rows, prompt, validate_metadata_membership, validate_record)
from .metrics import (aggregate_stage2, all_keywords_match, get_hop_category,
                      resolve_exposure_target, stage1_metrics, stage2_em)
from .utils import add_common_args, fresh_output, save_csv, save_json, seed_everything

def infer(model, processor, row, max_new_tokens):
    import torch
    inputs = processor(text=prompt(row['question']), images=row['image'], return_tensors='pt')
    inputs = inputs.to(model.device, model.dtype)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    prediction = processor.decode(output[0], skip_special_tokens=True).split('ASSISTANT:')[-1].strip()
    return {'image_path': row['image_path'], 'question': row['question'], 'ground_truth': row['answer'],
            'answer': row['answer'], 'prediction': prediction, 'keywords': row['keywords'],
            'keywords_raw': row['keywords_raw'], 'keyword': row['keywords_raw'],
            'question_type': row['question_type'], 'qa_category': row['qa_category'],
            'attribute': row['attribute'], 'cloze_prompt': row['cloze_prompt'],
            'hop': get_hop_category(row['question_type'])}

def stage1_perplexity(model, processor, row):
    # Preserve eval_finetune.py concatenation, which has NO extra space.
    import torch
    text = prompt(row['question'])
    inputs = processor(text=text + row['answer'], images=row['image'], return_tensors='pt').to(model.device, model.dtype)
    prompt_len = processor(text=text, images=row['image'], return_tensors='pt').input_ids.shape[1]
    labels = inputs.input_ids.clone()
    labels[:, :min(prompt_len, labels.shape[1])] = -100
    labels[labels == -200] = -100
    with torch.no_grad():
        outputs = model(input_ids=inputs.input_ids, pixel_values=inputs.pixel_values,
                        attention_mask=inputs.attention_mask, labels=labels)
    return torch.exp(outputs.loss).item()

def select_stage2_roles(metadata, ratio):
    names = get_forget_identities(ratio)
    for split in ('finetune', 'retain', 'test', forget_split(ratio)):
        validate_metadata_membership(metadata[split], split)
    if {identity(r) for r in metadata[forget_split(ratio)]} != names:
        raise ValueError('Forget identity manifest does not match the published split')
    return {'forget': ('finetune', [i for i, r in enumerate(metadata['finetune']) if identity(r) in names]),
            'retain': ('retain', [i for i, r in enumerate(metadata['retain']) if identity(r) not in names]),
            'test': ('test', [i for i, r in enumerate(metadata['test']) if identity(r) in names]),
            'exposure': ('retain', [i for i, r in enumerate(metadata['retain']) if identity(r) in names])}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument('--stage', type=int, choices=[1, 2], required=True)
    parser.add_argument('--finetuned-adapter', required=True)
    parser.add_argument('--unlearn-adapter')
    parser.add_argument('--split', choices=list(SPLITS), default='retain', help='Stage 1 split')
    parser.add_argument('--forget-ratio', type=int, choices=[5, 10, 15, 20], default=20)
    parser.add_argument('--precision', choices=['fp16', 'bf16', 'fp32'], default='fp16')
    parser.add_argument('--max-new-tokens', type=int, default=100)
    parser.add_argument('--output-dir', default='outputs/evaluation')
    parser.add_argument('--limit', type=int, help='Smoke subset per role; cannot be reported as full benchmark')
    parser.add_argument('--exposure', action='store_true', help='Score all original attribute candidates (many forward passes)')
    parser.add_argument('--gpt-score', action='store_true', help='Optional Gemini API judging; requires GOOGLE_API_KEY')
    parser.add_argument('--judge-model', default='gemini-2.0-flash')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if args.stage == 2 and not args.unlearn_adapter:
        parser.error('Stage 2 requires --unlearn-adapter')
    if args.stage == 1 and (args.exposure or args.gpt_score):
        parser.error('--exposure and --gpt-score follow the original Stage 2 protocol only')
    from .model import load_evaluation
    from rouge_score import rouge_scorer
    seed_everything(args.seed)
    output = fresh_output(args.output_dir)
    judge = None
    if args.gpt_score:
        from .judge import GeminiJudge
        judge = GeminiJudge(model_name=args.judge_model)
    datasets = {}
    if args.stage == 1:
        datasets[args.split] = load_split(args.split, revision=args.revision, cache_dir=args.cache_dir,
                                         streaming=args.limit is not None)
        roles = {'stage1': (args.split, None)}
    else:
        for split in ('finetune', 'retain', 'test', forget_split(args.forget_ratio)):
            datasets[split] = load_split(split, revision=args.revision, cache_dir=args.cache_dir)
        roles = select_stage2_roles({s: metadata_rows(ds) for s, ds in datasets.items()}, args.forget_ratio)
    model, processor = load_evaluation(args)
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    detailed, expected_counts = {}, {}
    for role, (split, indices) in roles.items():
        expected_counts[role] = len(indices) if indices is not None else SPLITS[split]
        if role == 'exposure' and not args.exposure:
            continue
        rows = (datasets[split][i] for i in indices) if indices is not None else iter(datasets[split])
        if args.limit:
            rows = itertools.islice(rows, args.limit)
        detailed[role] = []
        for sample in rows:
            row = validate_record(sample)
            if role == 'exposure':
                from .exposure import calculate_exposure_score
                target, candidates = resolve_exposure_target(row)
                score = calculate_exposure_score(model, processor, row['image'], row['cloze_prompt'], target, candidates)
                detailed[role].append({'image_path': row['image_path'], 'attribute': row['attribute'],
                                       'ground_truth': target, 'candidates': candidates, 'cloze_prompt': row['cloze_prompt'],
                                       'exposure': score, 'hop': get_hop_category(row['question_type'])})
                continue
            result = infer(model, processor, row, args.max_new_tokens)
            if args.stage == 1:
                result['perplexity'] = stage1_perplexity(model, processor, row)
            else:
                result['em'] = stage2_em(row['keywords_raw'], result['prediction'])
                result['all_keywords_match_diagnostic'] = bool(all_keywords_match(row['keywords'], result['prediction']))
                if role == 'retain':
                    result['rougeL'] = rouge.score(row['answer'], result['prediction'])['rougeL'].fmeasure
                if role == 'forget' and judge:
                    result['gpt_score'], result['gpt_error'] = judge.evaluate(result['keyword'], result['prediction'])
            detailed[role].append(result)
        # Write each completed role so its predictions survive later-role failures.
        save_json(output / f'{role}_predictions.json', detailed[role])
        save_csv(output / f'{role}_predictions.csv', detailed[role])
    if args.stage == 1:
        detailed['stage1'], summary = stage1_metrics(detailed['stage1'])
        save_json(output / 'stage1_predictions.json', detailed['stage1'])
        save_csv(output / 'stage1_predictions.csv', detailed['stage1'])
    else:
        summary = aggregate_stage2(detailed)
        if judge:
            summary['gpt_score'] = {}
            for label, hop in [('overall', None), ('single_hop', 'single-hop'), ('multi_hop', 'multi-hop')]:
                scores = [r['gpt_score'] for r in detailed['forget'] if r['gpt_score'] is not None and (hop is None or r['hop'] == hop)]
                summary['gpt_score'][label] = round(sum(scores) / len(scores) / 100, 4) if scores else None
            summary['gpt_failures'] = sum(r['gpt_score'] is None for r in detailed['forget'])
    save_json(output / 'results.json', {'arguments': vars(args), 'is_smoke_subset': args.limit is not None,
                                      'expected_counts': expected_counts, 'evaluated_counts': {k: len(v) for k, v in detailed.items()},
                                      'summary': summary, 'detailed_results': detailed})
    save_json(output / 'metrics.json', summary)
    save_csv(output / 'metrics.csv', [{'metric': k, **v} if isinstance(v, dict) else {'metric': k, 'value': v}
                                     for k, v in summary.items()])
    print(summary)

if __name__ == '__main__':
    main()
