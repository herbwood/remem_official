"""CPU tests for benchmark semantics, masking, and the copied loss functions."""
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from PIL import Image

from remem.data import (CONFIG_DIR, MemorizationDataset, UnlearningDataset, forget_split,
                        identity, parse_keywords, process_item, prompt, validate_record)
from remem.losses import (DPOUnlearningTrainer, GradientAscentTrainer, GradientDifferenceTrainer,
                          KLDivergenceTrainer, NPOUnlearningTrainer, get_batch_logps,
                          MultimodalUnlearningCollator)
from remem.collator import LlavaDataCollator
from remem.metrics import (aggregate_stage2, all_keywords_match, resolve_exposure_target,
                           stage1_metrics, stage2_em)
from remem.exposure import calculate_exposure_score
from remem.utils import clean_json, fresh_output, save_csv, save_json

def sample():
    return {'image': Image.new('RGB', (16, 16)), 'question': 'What is the salary?',
            'answer': "Aiko Tanaka's annual salary usd is 29,500.", 'keywords': "['29,500']",
            'question_type': '1-hop', 'qa_category': 'personal_information',
            'attribute': 'annual_salary_usd', 'cloze_prompt': 'The annual salary usd of the person in the image is',
            'image_path': 'Aiko_Tanaka_91.png'}

class TinyProcessor:
    """Separately controllable prompt/answer token lengths and padding."""
    def __init__(self):
        self.tokenizer = self
        self.pad_token_id = 0
        self.calls = []
    def __call__(self, *, text, images, return_tensors):
        self.calls.append(text)
        ids = [1, 2, 3] if text.endswith('ASSISTANT:') else [1, 2, 3, 4, 5]
        return {'input_ids': torch.tensor([ids]), 'attention_mask': torch.ones((1, len(ids)), dtype=torch.long),
                'pixel_values': torch.zeros((1, 3, 4, 4))}
    def pad(self, data, padding=True, return_tensors='pt'):
        values = data['input_ids']
        length = max(len(v) for v in values)
        ids = torch.zeros((len(values), length), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, value in enumerate(values):
            ids[i, :len(value)] = torch.as_tensor(value)
            mask[i, :len(value)] = 1
        return {'input_ids': ids, 'attention_mask': mask}

class ConstantModel:
    def __init__(self, loss, logits=None):
        self.loss = torch.tensor(loss, requires_grad=True)
        self.logits = torch.tensor([[[0.1, 0.9], [0.6, 0.4], [0.3, 0.7]]], requires_grad=True) if logits is None else logits
        self.device = torch.device('cpu')
    def eval(self):
        return self
    def __call__(self, **batch):
        return SimpleNamespace(loss=self.loss, logits=self.logits + 0)

class DataTests(unittest.TestCase):
    def test_published_keyword_types_and_rejection(self):
        self.assertEqual(parse_keywords("['555', '857', '3669']"), ['555', '857', '3669'])
        self.assertEqual(parse_keywords('aiko.tanaka@example.com'), ['aiko.tanaka@example.com'])
        self.assertEqual(parse_keywords(['Invests', 'stocks']), ['Invests', 'stocks'])
        for bad in ('', '[]', '[malicious()]', [1], None):
            with self.assertRaises(ValueError):
                parse_keywords(bad)
    def test_images_and_metadata(self):
        row = validate_record(sample())
        self.assertIsInstance(row['image'], Image.Image)
        self.assertEqual(row['keywords'], ['29,500'])
        for hop in ('1-hop', 'multi-hop'):
            self.assertEqual(validate_record({**sample(), 'question_type': hop})['question_type'], hop)
        for update in ({'image': 'private.png'}, {'question': ''}, {'answer': ''}, {'question_type': 'unknown'}):
            with self.assertRaises(ValueError):
                validate_record({**sample(), **update})
        self.assertEqual(identity({'image_path': 'Maya_Sharma_Rodriguez_test_5.png'}), 'Maya_Sharma_Rodriguez')
        self.assertEqual(identity({'image_path': 'Maya_Sharma_Rodriguez_5.png'}), 'Maya_Sharma_Rodriguez')
    def test_all_forget_ratios_and_non_nested_sets(self):
        self.assertEqual([forget_split(i) for i in (5, 10, 15, 20)], ['forget1', 'forget2', 'forget3', 'forget4'])
        with self.assertRaises(ValueError):
            forget_split(25)
        names = json.loads((CONFIG_DIR / 'forget_identities.json').read_text())
        self.assertFalse(set(names['2']).issubset(names['3']))
    def test_answer_only_mask_and_collator_padding(self):
        processor = TinyProcessor()
        output = process_item(processor, sample())
        self.assertEqual(output['labels'].tolist(), [-100, -100, -100, 4, 5])
        self.assertEqual(processor.calls[0], prompt(sample()['question']) + ' ' + sample()['answer'])
        shorter = {**output, 'input_ids': output['input_ids'][:4], 'labels': output['labels'][:4],
                   'attention_mask': output['attention_mask'][:4]}
        for collator in (LlavaDataCollator(processor), MultimodalUnlearningCollator(processor)):
            batch = collator([output, shorter]) if isinstance(collator, LlavaDataCollator) else collator._batch_samples([output, shorter])
            self.assertEqual(batch['labels'][1, -1].item(), -100)
            self.assertEqual(tuple(batch['pixel_values'].shape), (2, 3, 4, 4))
    def test_unlearning_pairing_and_no_test_inputs(self):
        rows = [sample(), {**sample(), 'image_path': 'Aisha_Williams_1.png'}]
        processor = TinyProcessor()
        for method in ('GA', 'GD', 'KL', 'DPO', 'NPO'):
            dataset = UnlearningDataset(rows, [0], [1], processor, method, ['I do not know.'])
            self.assertEqual(len(dataset), 1)
            keys = set(dataset[0])
            expected = {'forget_sample'} if method == 'GA' else {'forget_sample', 'idk_sample'} if method in ('DPO', 'NPO') else {'forget_sample', 'retain_sample'}
            self.assertEqual(keys, expected)
        with self.assertRaises(ValueError):
            UnlearningDataset(rows, [0], [1], processor, 'DPO')

class MetricsTests(unittest.TestCase):
    def test_fractional_keyword_score_rouge_and_hop_aggregation(self):
        rows = [{'ground_truth': 'Aiko Tanaka', 'prediction': 'Aiko', 'keywords': ['Aiko', 'Tanaka'],
                 'question_type': '1-hop', 'qa_category': 'identity'},
                {'ground_truth': '555-857-3669', 'prediction': '555-857-3669', 'keywords': ['555', '857', '3669'],
                 'question_type': 'multi-hop', 'qa_category': 'personal_information'}]
        detailed, metrics = stage1_metrics(rows)
        self.assertEqual(metrics['overall']['keyword_score'], 75.0)
        self.assertEqual(detailed[1]['rougeL_f1'], 1.0)
        self.assertEqual(metrics['categories']['1-hop_identity']['keyword_score'], 50.0)
    def test_source_stage2_em_differs_from_keyword_diagnostic(self):
        # Source coercion of a list leaves spaces between date/phone components.
        self.assertFalse(stage2_em("['555', '857', '3669']", '555-857-3669'))
        self.assertTrue(all_keywords_match(['555', '857', '3669'], '555-857-3669'))
        self.assertTrue(stage2_em('555-857-3669', 'The number is 555-857-3669.'))
        self.assertTrue(stage2_em("['Aiko', 'Tanaka']", 'Aiko Tanaka is pictured.'))
    def test_exposure_candidate_ranking_and_stable_ties(self):
        candidates = ['a', 'b', 'c']
        with patch('remem.exposure.calculate_ppl', side_effect=[2, 1, 3]):
            self.assertEqual(calculate_exposure_score(None, None, None, 'cloze', 'b', candidates), 1.0)
        with patch('remem.exposure.calculate_ppl', return_value=2):
            self.assertEqual(calculate_exposure_score(None, None, None, 'cloze', 'b', candidates), 0.5)
        self.assertTrue(math.isnan(calculate_exposure_score(None, None, None, 'cloze', 'z', candidates)))
        target, values = resolve_exposure_target(validate_record(sample()))
        self.assertEqual(target, '29,500')
        self.assertEqual(len(values), 20)
        with self.assertRaises(ValueError):
            resolve_exposure_target(validate_record({**sample(), 'keywords': "['99,999']"}))
    def test_aggregation_missing_groups_and_portable_outputs(self):
        summary = aggregate_stage2({'forget': [{'em': True, 'hop': 'single-hop'}, {'em': False, 'hop': 'multi-hop'}]})
        self.assertEqual(summary['forget_em']['overall'], 0.5)
        self.assertEqual(summary['forget_em']['count'], 2)
        self.assertIsNone(summary['test_em']['overall'])
        with tempfile.TemporaryDirectory() as directory:
            path = fresh_output(Path(directory) / 'run')
            save_json(path / 'metrics.json', {'missing': float('nan')})
            save_csv(path / 'predictions.csv', [{'keywords': ['Aiko', 'Tanaka']}])
            self.assertIsNone(json.loads((path / 'metrics.json').read_text())['missing'])
            with self.assertRaises(ValueError):
                fresh_output(path)

class LossTests(unittest.TestCase):
    def test_logps_ignore_prompt_and_sum_answer_tokens(self):
        logits = torch.tensor([[[0., 1.], [2., 0.], [0., 0.]]], requires_grad=True)
        labels = torch.tensor([[-100, 1, -100]])
        logp = get_batch_logps(logits, labels)
        self.assertTrue(torch.allclose(logp, F.log_softmax(logits, -1)[:, 0, 1]))
        logp.sum().backward()
        self.assertEqual(logits.grad[:, 1:].abs().sum().item(), 0)
    def test_ga_and_gd_sign(self):
        model = ConstantModel(2.)
        inputs = {'forget_batch': {}, 'retain_batch': {}}
        ga = object.__new__(GradientAscentTrainer)
        self.assertEqual(ga.compute_loss(model, inputs).item(), -2.)
        self.assertEqual(ga.compute_loss(model, inputs, return_outputs=True)[0].item(), -2.)
        gd = object.__new__(GradientDifferenceTrainer)
        gd.gd_alpha = 0.5
        self.assertEqual(gd.compute_loss(model, inputs).item(), 1.)
    def test_kl_preserves_mean_over_all_logits(self):
        model = ConstantModel(2.)
        reference = ConstantModel(2., logits=torch.zeros((1, 3, 2)))
        trainer = object.__new__(KLDivergenceTrainer)
        trainer.model_ref, trainer.kl_beta = reference, 0.7
        actual = trainer.compute_loss(model, {'forget_batch': {}, 'retain_batch': {}})
        expected = -model.loss + 0.7 * F.kl_div(F.log_softmax(model.logits, -1), F.softmax(reference.logits, -1), reduction='mean')
        self.assertTrue(torch.allclose(actual, expected))
    def test_dpo_and_npo_reference_equal_policies(self):
        model, reference = ConstantModel(2.), ConstantModel(2.)
        batch = {'labels': torch.tensor([[-100, 1, 0]])}
        for cls in (DPOUnlearningTrainer, NPOUnlearningTrainer):
            trainer = object.__new__(cls)
            trainer.model_ref, trainer.beta = reference, 0.1
            loss = trainer.compute_loss(model, {'forget_batch': batch, 'idk_batch': batch})
            self.assertAlmostEqual(loss.item(), math.log(2), places=6)
            loss.backward()
            self.assertIsNone(reference.logits.grad)

if __name__ == '__main__':
    unittest.main()
