"""Original HF LLaVA/PEFT loading sequence with portable device selection."""
from pathlib import Path

def check_adapter(path):
    if not path or not (Path(path) / 'adapter_config.json').is_file():
        raise FileNotFoundError(f'Explicit local PEFT adapter required: {path}')

def dtype_for(name):
    import torch
    return {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[name]

def configure_processor(processor, config, *, training=False):
    # Transformers 4.46 expands the image tokens in the processor. Match the
    # model config so original answer masking and DPO logit/label alignment work.
    processor.patch_size = config.vision_config.patch_size
    processor.vision_feature_select_strategy = config.vision_feature_select_strategy
    processor.num_additional_image_tokens = 1  # CLIP vision encoder CLS token.
    if training:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return processor

def load_base(model_id, *, precision='fp16', device_map='auto', quantization='none',
              revision='main', cache_dir=None, training=False):
    import torch
    from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
    options = {'device_map': device_map, 'revision': revision, 'cache_dir': cache_dir}
    if quantization == '4bit':
        if not torch.cuda.is_available():
            raise RuntimeError('Original Stage 1 4-bit loading requires a compatible CUDA/bitsandbytes host.')
        options['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True,
                                                          bnb_4bit_compute_dtype=torch.float16)
    else:
        options['torch_dtype'] = dtype_for(precision)
    model = LlavaForConditionalGeneration.from_pretrained(model_id, **options)
    processor = AutoProcessor.from_pretrained(model_id, revision=revision, cache_dir=cache_dir)
    return model, configure_processor(processor, model.config, training=training)

def attach_lora(model, config):
    from peft import LoraConfig, get_peft_model
    return get_peft_model(model, LoraConfig(**config, task_type='CAUSAL_LM'))

def load_memorized(args, *, precision='fp16', merge=False, training=False, device_map=None):
    from peft import PeftModel
    check_adapter(args.finetuned_adapter)
    model, processor = load_base(args.model_id, precision=precision,
                                 device_map=device_map or args.device_map,
                                 revision=args.model_revision, cache_dir=args.cache_dir,
                                 training=training)
    model = PeftModel.from_pretrained(model, args.finetuned_adapter)
    if merge:
        model = model.merge_and_unload()
    return model, processor

def load_evaluation(args):
    from peft import PeftModel
    if args.stage == 2:
        check_adapter(args.unlearn_adapter)
    model, processor = load_memorized(args, precision=args.precision, merge=args.stage == 2)
    if args.stage == 2:
        model = PeftModel.from_pretrained(model, args.unlearn_adapter)
    model.eval()
    return model, processor
