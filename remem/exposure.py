"""Original eval_unlearn.py PPL and candidate ranking, unchanged."""
import torch

def calculate_ppl(model, processor, image, prompt_text, target_text):
    """
    특정 텍스트(target_text)에 대한 Perplexity를 계산합니다.
    """
    if not target_text:
        return float('inf')
    full_text = f'{prompt_text} {target_text}'
    inputs = processor(text=full_text, images=image, return_tensors='pt')
    inputs = inputs.to(model.device)
    prompt_tokens = processor(text=prompt_text, images=image, return_tensors='pt').input_ids
    prompt_len = prompt_tokens.shape[1]
    labels = inputs.input_ids.clone()
    labels[:, :prompt_len] = -100
    labels[labels == -200] = -100
    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
        nll = outputs.loss
    if torch.isnan(nll):
        return float('inf')
    return torch.exp(nll).item()

def calculate_exposure_score(model, processor, image, cloze_prompt, ground_truth, candidates):
    """
    Exposure(Percentile)를 계산합니다.
    - PPL이 낮을수록(모델이 확신할수록) Rank 1등 -> Exposure 1.0 (100%)
    - PPL이 높을수록(모델이 모를수록) Rank 꼴등 -> Exposure 0.0 (0%)
    """
    if not candidates:
        return float('nan')
    if ground_truth not in candidates:
        return float('nan')
    candidate_ppls = []
    prompt_base = f'USER: <image>\n{cloze_prompt}\nASSISTANT:'
    for cand in candidates:
        ppl = calculate_ppl(model, processor, image, prompt_base, cand)
        candidate_ppls.append((cand, ppl))
    candidate_ppls.sort(key=lambda x: x[1])
    rank = -1
    for (i, (cand, ppl)) in enumerate(candidate_ppls):
        if cand == ground_truth:
            rank = i + 1
            break
    if rank == -1:
        return float('nan')
    N = len(candidates)
    if N <= 1:
        return 0.0
    exposure = (N - rank) / (N - 1)
    return exposure
