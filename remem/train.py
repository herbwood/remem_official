"""Stage 1: finetune.py training flow with HF images and original LoRA config."""
from .data import MemorizationDataset, load_split, metadata_rows, validate_metadata_membership
from .utils import (fresh_output, read_config, report_effective_batch_size, save_json, seed_everything,
                    training_arguments, training_parser)

def main():
    parser = training_parser(1)
    parser.add_argument('--quantization', choices=['4bit', 'none'], default='4bit')
    args = parser.parse_args()
    from transformers import Trainer
    from .collator import LlavaDataCollator
    from .model import attach_lora, load_base
    config = read_config(args)
    seed_everything(args.seed)
    ds = load_split('finetune', revision=args.revision, cache_dir=args.cache_dir)
    validate_metadata_membership(metadata_rows(ds), 'finetune')
    output = fresh_output(args.output_dir)
    precision = 'bf16' if config['training'].get('bf16') else 'fp16' if config['training'].get('fp16') else 'fp32'
    model, processor = load_base(args.model_id, precision=precision, quantization=args.quantization,
                                 device_map=args.device_map, revision=args.model_revision,
                                 cache_dir=args.cache_dir, training=True)
    model = attach_lora(model, config['lora'])
    model.print_trainable_parameters()
    training_args = training_arguments(args, config, output)
    trainer = Trainer(model=model, args=training_args, train_dataset=MemorizationDataset(ds, processor),
                      data_collator=LlavaDataCollator(processor))
    effective = report_effective_batch_size(training_args)
    save_json(output / 'run.json', {'arguments': {k: str(v) if k == 'config' else v for k, v in vars(args).items()},
                                  'config': config, 'effective_batch_size': effective,
                                  'training_arguments': training_args.to_dict()})
    trainer.train()
    model.save_pretrained(output / 'final_adapter')
    processor.save_pretrained(output / 'final_adapter')

if __name__ == '__main__':
    main()
