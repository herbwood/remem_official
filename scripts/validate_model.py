"""CPU forward/backward with a random tiny LLaVA; fetch processor/config only."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', default='.cache')
    parser.add_argument('--output', default='validation/tiny_model.json')
    args = parser.parse_args()
    import torch
    from PIL import Image
    from transformers import AutoConfig, AutoProcessor, LlavaConfig, LlavaForConditionalGeneration
    from remem.data import process_item
    from remem.model import attach_lora, configure_processor
    from remem.collator import LlavaDataCollator
    from remem.losses import get_batch_logps
    from remem.utils import save_json
    import json
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model_id = 'llava-hf/llava-1.5-7b-hf'
    published = AutoConfig.from_pretrained(model_id, cache_dir=args.cache_dir)
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=args.cache_dir)
    text_config = {'model_type': 'llama', 'vocab_size': published.text_config.vocab_size,
                   'hidden_size': 16, 'intermediate_size': 32, 'num_hidden_layers': 1,
                   'num_attention_heads': 2, 'num_key_value_heads': 2, 'max_position_embeddings': 2048}
    vision_config = {'model_type': 'clip_vision_model', 'hidden_size': 16, 'intermediate_size': 32,
                     'num_hidden_layers': 2, 'num_attention_heads': 2,
                     'image_size': published.vision_config.image_size,
                     'patch_size': published.vision_config.patch_size}
    config = LlavaConfig(text_config=text_config, vision_config=vision_config,
                         image_token_index=published.image_token_index,
                         vision_feature_layer=published.vision_feature_layer,
                         vision_feature_select_strategy=published.vision_feature_select_strategy)
    model = LlavaForConditionalGeneration(config)
    processor = configure_processor(processor, config, training=True)
    original_lora = json.loads((ROOT / 'configs/stage1.json').read_text())['lora']
    model = attach_lora(model, {**original_lora, 'r': 2, 'lora_alpha': 4})
    row = {'image': Image.new('RGB', (32, 32)), 'image_path': 'Aiko_Tanaka_91.png',
           'question': 'What is the annual salary?', 'answer': '29,500.', 'keywords': "['29,500']",
           'question_type': '1-hop', 'qa_category': 'personal_information', 'attribute': 'annual_salary_usd',
           'cloze_prompt': 'The annual salary usd of the person in the image is'}
    item = process_item(processor, row)
    batch = LlavaDataCollator(processor)([item])
    result = model(**batch)
    assert result.logits.shape[:2] == batch['labels'].shape
    assert torch.isfinite(result.loss)
    logps = get_batch_logps(result.logits.float(), batch['labels'])
    assert torch.isfinite(logps).all()
    result.loss.backward()
    gradients = [param.grad for param in model.parameters() if param.requires_grad and param.grad is not None]
    assert gradients and all(torch.isfinite(grad).all() for grad in gradients)
    image_tokens = int((batch['input_ids'] == config.image_token_index).sum())
    assert image_tokens == 576
    report = {'model': 'random tiny LLaVA; no pretrained weights downloaded',
              'processor': model_id, 'image_tokens': image_tokens,
              'logit_label_alignment': True, 'finite_loss': True, 'finite_lora_gradients': True,
              'original_target_module_list_accepted': True, 'trainable_gradient_tensors': len(gradients)}
    save_json(ROOT / args.output, report)
    print(report)

if __name__ == '__main__':
    main()
