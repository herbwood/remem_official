"""Stage 2: original five trainers, verified identity selection, fresh LoRA."""
import json
from pathlib import Path
from .data import (UnlearningDataset, forget_split, load_split, metadata_rows, stage2_partition)
from .utils import (fresh_output, read_config, report_effective_batch_size, save_json, seed_everything,
                    training_arguments, training_parser)

def main():
    parser = training_parser(2)
    parser.add_argument('--method', choices=['GA', 'GD', 'KL', 'DPO', 'NPO'], required=True)
    parser.add_argument('--forget-ratio', type=int, choices=[5, 10, 15, 20], default=20)
    parser.add_argument('--finetuned-adapter', required=True)
    parser.add_argument('--reference-device', default='auto', help='Device placement for frozen reference model')
    parser.add_argument('--reference-precision', choices=['fp16', 'bf16', 'fp32'], default='fp16')
    parser.add_argument('--idk-responses', type=Path, help='Original JSON object containing an idk list (DPO/NPO)')
    args = parser.parse_args()
    from .losses import (MultimodalUnlearningCollator, GradientAscentTrainer, GradientDifferenceTrainer,
                         KLDivergenceTrainer, DPOUnlearningTrainer, NPOUnlearningTrainer)
    from .model import attach_lora, check_adapter, load_memorized
    check_adapter(args.finetuned_adapter)
    idk = None
    if args.method in ('DPO', 'NPO'):
        if args.idk_responses is None:
            parser.error('DPO/NPO require --idk-responses: original idk.json is absent from this repository.')
        idk = json.loads(args.idk_responses.read_text(encoding='utf-8')).get('idk')
        if not isinstance(idk, list) or not idk or any(not isinstance(s, str) or not s.strip() for s in idk):
            parser.error('--idk-responses must contain a nonempty idk list of nonempty strings')
    config = read_config(args)
    seed_everything(args.seed)
    ds = load_split('finetune', revision=args.revision, cache_dir=args.cache_dir)
    published = load_split(forget_split(args.forget_ratio), revision=args.revision,
                            cache_dir=args.cache_dir, streaming=True)
    forget, retain = stage2_partition(metadata_rows(ds), metadata_rows(published), args.forget_ratio)
    output = fresh_output(args.output_dir)
    precision = 'bf16' if config['training'].get('bf16') else 'fp16' if config['training'].get('fp16') else 'fp32'
    model, processor = load_memorized(args, precision=precision, merge=True, training=True)
    model = attach_lora(model, config['lora'])
    model.print_trainable_parameters()
    reference = None
    if args.method in ('KL', 'DPO', 'NPO'):
        reference, _ = load_memorized(args, precision=args.reference_precision,
                                     device_map=args.reference_device, training=True)
        reference.eval()
        for param in reference.parameters():
            param.requires_grad = False
    trainers = {'GA': (GradientAscentTrainer, {}),
                'GD': (GradientDifferenceTrainer, {'gd_alpha': config['params']['gd_alpha']}),
                'KL': (KLDivergenceTrainer, {'model_ref': reference, 'kl_beta': config['params']['kl_beta']}),
                'DPO': (DPOUnlearningTrainer, {'model_ref': reference, 'beta': config['params']['dpo_beta']}),
                'NPO': (NPOUnlearningTrainer, {'model_ref': reference, 'beta': config['params']['dpo_beta']})}
    trainer_class, options = trainers[args.method]
    training_args = training_arguments(args, config, output)
    dataset = UnlearningDataset(ds, forget, retain, processor, args.method, idk)
    trainer = trainer_class(model=model, args=training_args, train_dataset=dataset,
                            data_collator=MultimodalUnlearningCollator(processor), **options)
    effective = report_effective_batch_size(training_args)
    save_json(output / 'run.json', {'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                                  'config': config, 'forget_train': len(forget), 'retain_train': len(retain),
                                  'effective_batch_size': effective,
                                  'training_arguments': training_args.to_dict()})
    trainer.train()
    model.save_pretrained(output / 'final_unlearn_adapter')
    processor.save_pretrained(output / 'final_unlearn_adapter')

if __name__ == '__main__':
    main()
