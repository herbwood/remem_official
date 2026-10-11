"""Portable configuration and output handling."""
import argparse
import csv
import json
import math
import random
from pathlib import Path
from .data import DEFAULT_REVISION

def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def clean_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    return value

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(value), indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')

def fresh_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f'Output directory is not empty: {path}; select a new directory.')
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_csv(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in records for k in row))
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in records:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                             for k, v in clean_json(row).items()})

def add_common_args(parser):
    parser.add_argument('--model-id', default='llava-hf/llava-1.5-7b-hf')
    parser.add_argument('--model-revision', default='main')
    parser.add_argument('--revision', default=DEFAULT_REVISION, help='Hugging Face dataset revision')
    parser.add_argument('--cache-dir')
    parser.add_argument('--device-map', default='auto', help='auto, cpu, cuda, or an explicit device')
    parser.add_argument('--seed', type=int, default=42)

def training_parser(stage):
    parser = argparse.ArgumentParser(description=f'ReMem Stage {stage} (original Trainer protocol).')
    add_common_args(parser)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'configs' / f'stage{stage}.json')
    parser.add_argument('--output-dir', default=f'outputs/stage{stage}')
    parser.add_argument('--epochs', type=float)
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--gradient-accumulation', type=int)
    parser.add_argument('--precision', choices=['bf16', 'fp16', 'fp32'])
    parser.add_argument('--optimizer', choices=['adamw_torch', 'adamw_hf'])
    return parser

def read_config(args):
    config = json.loads(args.config.read_text(encoding='utf-8'))
    names = {'epochs': 'num_train_epochs', 'learning_rate': 'learning_rate',
             'batch_size': 'per_device_train_batch_size', 'gradient_accumulation': 'gradient_accumulation_steps',
             'optimizer': 'optim'}
    for arg, key in names.items():
        if getattr(args, arg, None) is not None:
            config['training'][key] = getattr(args, arg)
    if args.precision:
        config['training']['bf16'] = args.precision == 'bf16'
        config['training']['fp16'] = args.precision == 'fp16'
    if config['training']['per_device_train_batch_size'] < 1 or config['training']['gradient_accumulation_steps'] < 1:
        raise ValueError('Batch size and gradient accumulation must be positive')
    return config

def training_arguments(args, config, output):
    from transformers import TrainingArguments
    training = dict(config['training'])
    training.update(output_dir=str(output), seed=args.seed, data_seed=args.seed,
                    remove_unused_columns=False, report_to='none')
    result = TrainingArguments(**training)
    return result

def report_effective_batch_size(training_args):
    # Query after Trainer has resolved model/data-parallel placement.
    effective = training_args.train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
    print(f'Effective global batch size: {effective}')
    if effective != 64:
        print('This run differs from the paper global batch size of 64.')
    return effective
