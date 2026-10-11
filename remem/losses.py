"""Unlearning losses from the original dataloader.py implementation."""
import torch
import torch.nn.functional as F
from transformers import Trainer
from typing import Dict, Any, List

class MultimodalUnlearningCollator:

    def __init__(self, processor):
        self.processor = processor

    def _batch_samples(self, samples: List[Dict]) -> Dict[str, torch.Tensor]:
        if not samples:
            return {}
        pixel_values_list = [s['pixel_values'] for s in samples]
        input_ids_list = [s['input_ids'] for s in samples]
        labels_list = [s['labels'] for s in samples]
        padded_inputs = self.processor.tokenizer.pad({'input_ids': input_ids_list}, return_tensors='pt', padding=True)
        padded_labels = self.processor.tokenizer.pad({'input_ids': labels_list}, return_tensors='pt', padding=True)
        label_input_ids = padded_labels['input_ids']
        label_input_ids[label_input_ids == self.processor.tokenizer.pad_token_id] = -100
        return {'pixel_values': torch.stack(pixel_values_list), 'input_ids': padded_inputs['input_ids'], 'attention_mask': padded_inputs['attention_mask'], 'labels': label_input_ids}

    def __call__(self, features: List[Dict]) -> Dict[str, Any]:
        features = [f for f in features if f is not None]
        if not features:
            return {}
        first_item_keys = features[0].keys()
        if 'idk_sample' in first_item_keys:
            idk_samples = [item['idk_sample'] for item in features]
            forget_samples = [item['forget_sample'] for item in features]
            return {'idk_batch': self._batch_samples(idk_samples), 'forget_batch': self._batch_samples(forget_samples)}
        elif 'retain_sample' in first_item_keys:
            forget_samples = [item['forget_sample'] for item in features]
            retain_samples = [item['retain_sample'] for item in features]
            return {'forget_batch': self._batch_samples(forget_samples), 'retain_batch': self._batch_samples(retain_samples)}
        elif 'forget_sample' in first_item_keys:
            forget_samples = [item['forget_sample'] for item in features]
            return {'forget_batch': self._batch_samples(forget_samples)}
        else:
            raise ValueError(f'Collator가 예상치 못한 데이터 형식을 받았습니다. Keys: {first_item_keys}')

def get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels = labels[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
    per_token_loss = loss_fct(shifted_logits.view(-1, shifted_logits.size(-1)), shifted_labels.view(-1))
    per_sequence_loss = per_token_loss.view(shifted_logits.size(0), -1).sum(dim=1)
    return -per_sequence_loss

class GradientAscentTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs['forget_batch'])
        return (-outputs.loss, outputs) if return_outputs else -outputs.loss

class GradientDifferenceTrainer(Trainer):

    def __init__(self, *args, gd_alpha=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.gd_alpha = gd_alpha

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        forget_loss = model(**inputs['forget_batch']).loss
        retain_loss = model(**inputs['retain_batch']).loss
        loss = retain_loss - self.gd_alpha * forget_loss
        return (loss, None) if return_outputs else loss

class KLDivergenceTrainer(Trainer):

    def __init__(self, *args, model_ref=None, kl_beta=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_ref = model_ref.eval()
        self.kl_beta = kl_beta

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        forget_loss = model(**inputs['forget_batch']).loss
        ref_device = self.model_ref.device
        retain_batch_ref_device = {k: v.to(ref_device) for (k, v) in inputs['retain_batch'].items()}
        with torch.no_grad():
            teacher_logits = self.model_ref(**retain_batch_ref_device).logits
        student_logits = model(**inputs['retain_batch']).logits
        student_logits_fp32 = student_logits.to(torch.float32)
        teacher_logits_fp32 = teacher_logits.to(student_logits.device, dtype=torch.float32)
        kl_loss = F.kl_div(F.log_softmax(student_logits_fp32, dim=-1), F.softmax(teacher_logits_fp32, dim=-1), reduction='mean', log_target=False)
        loss = -forget_loss + self.kl_beta * kl_loss
        return (loss, None) if return_outputs else loss

class DPOUnlearningTrainer(Trainer):

    def __init__(self, *args, model_ref=None, beta=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_ref = model_ref.eval()
        self.beta = beta

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        policy_chosen_outputs = model(**inputs['idk_batch'])
        policy_rejected_outputs = model(**inputs['forget_batch'])
        ref_device = self.model_ref.device
        idk_batch_ref_device = {k: v.to(ref_device) for (k, v) in inputs['idk_batch'].items()}
        forget_batch_ref_device = {k: v.to(ref_device) for (k, v) in inputs['forget_batch'].items()}
        with torch.no_grad():
            ref_chosen_outputs = self.model_ref(**idk_batch_ref_device)
            ref_rejected_outputs = self.model_ref(**forget_batch_ref_device)
        policy_chosen_logits_fp32 = policy_chosen_outputs.logits.to(torch.float32)
        policy_rejected_logits_fp32 = policy_rejected_outputs.logits.to(torch.float32)
        student_device = policy_chosen_logits_fp32.device
        ref_chosen_logits_fp32 = ref_chosen_outputs.logits.to(device=student_device, dtype=torch.float32)
        ref_rejected_logits_fp32 = ref_rejected_outputs.logits.to(device=student_device, dtype=torch.float32)
        policy_chosen_logps = get_batch_logps(policy_chosen_logits_fp32, inputs['idk_batch']['labels'])
        policy_rejected_logps = get_batch_logps(policy_rejected_logits_fp32, inputs['forget_batch']['labels'])
        ref_chosen_logps = get_batch_logps(ref_chosen_logits_fp32, inputs['idk_batch']['labels'])
        ref_rejected_logps = get_batch_logps(ref_rejected_logits_fp32, inputs['forget_batch']['labels'])
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        loss = -F.logsigmoid(self.beta * (pi_logratios - ref_logratios)).mean()
        return (loss, None) if return_outputs else loss

class NPOUnlearningTrainer(Trainer):

    def __init__(self, *args, model_ref=None, beta=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_ref = model_ref.eval()
        self.beta = beta

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        policy_rejected_outputs = model(**inputs['forget_batch'])
        ref_device = self.model_ref.device
        forget_batch_ref_device = {k: v.to(ref_device) for (k, v) in inputs['forget_batch'].items()}
        with torch.no_grad():
            ref_rejected_outputs = self.model_ref(**forget_batch_ref_device)
        policy_rejected_logits_fp32 = policy_rejected_outputs.logits.to(torch.float32)
        student_device = policy_rejected_logits_fp32.device
        ref_rejected_logits_fp32 = ref_rejected_outputs.logits.to(device=student_device, dtype=torch.float32)
        policy_rejected_logps = get_batch_logps(policy_rejected_logits_fp32, inputs['forget_batch']['labels'])
        ref_rejected_logps = get_batch_logps(ref_rejected_logits_fp32, inputs['forget_batch']['labels'])
        log_ratio = policy_rejected_logps - ref_rejected_logps
        loss = -F.logsigmoid(-self.beta * log_ratio).mean()
        return (loss, None) if return_outputs else loss
