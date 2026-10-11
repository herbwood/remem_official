"""Original Stage 1 collator from data_module.py."""
import torch

class LlavaDataCollator:

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        features = [f for f in features if f is not None]
        if not features:
            return None
        pixel_values = torch.stack([f['pixel_values'] for f in features])
        input_ids_list = [f['input_ids'] for f in features]
        labels_list = [f['labels'] for f in features]
        attention_mask_list = [f['attention_mask'] for f in features]
        padded_inputs = self.processor.tokenizer.pad({'input_ids': input_ids_list}, padding='longest', return_tensors='pt')
        padded_labels = self.processor.tokenizer.pad({'input_ids': labels_list}, padding='longest', return_tensors='pt')
        padded_attention_mask = self.processor.tokenizer.pad({'input_ids': attention_mask_list}, padding='longest', return_tensors='pt')
        padded_labels['input_ids'][padded_labels['attention_mask'] == 0] = -100
        return {'pixel_values': pixel_values, 'input_ids': padded_inputs['input_ids'], 'attention_mask': padded_inputs['attention_mask'], 'labels': padded_labels['input_ids']}
