# ReMem

Official code and dataset for **Before Forgetting, Learn to Remember: Revisiting Foundational Learning Failures in LVLM Unlearning Benchmarks** (Findings of ACL 2026).

[Paper](https://aclanthology.org/2026.findings-acl.1701/) · [Dataset](https://huggingface.co/datasets/herbwood27/Remem)

ReMem examines foundational memorization before evaluating unlearning. It provides fictitious identities, multiple visual contexts, and single-hop and multi-hop QA. The code supports LLaVA-1.5-7B/13B with LoRA and five unlearning methods: GA, GD, KL, DPO, and NPO.

![ReMem overview](overview.png)

## Installation

Use Python 3.10 and run the following commands from this directory. Training requires a compatible CUDA environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1`.

## Dataset

The dataset is available on [Hugging Face](https://huggingface.co/datasets/herbwood27/Remem). It includes a `finetune` split for memorization, four forget splits (`forget1`–`forget4`) corresponding to 5%, 10%, 15%, and 20% of identities, and `retain` and `test` splits for evaluation.

Each sample contains an image, a question and answer, keywords, and metadata describing the question type, category, and attribute. Images are loaded directly as PIL images. To inspect a sample without downloading the entire dataset:

```python
from datasets import load_dataset
from remem.data import DEFAULT_REVISION, validate_record

dataset = load_dataset(
    "herbwood27/Remem",
    split="finetune",
    revision=DEFAULT_REVISION,
    streaming=True,
    cache_dir=".cache",
)
sample = next(iter(dataset))
image = sample["image"]
record = validate_record(sample)
print(record["question"], record["answer"], record["keywords"])
```

For unlearning, forget training examples are selected from `finetune` by identity; the remaining examples are used for retain training.

## Stage 1: memorization

```bash
python -m remem.train \
  --model-id llava-hf/llava-1.5-7b-hf \
  --output-dir outputs/memorize-7b --cache-dir .cache
```

For the 13B model, use `llava-hf/llava-1.5-13b-hf` and a separate output directory. The trained adapter is saved in `final_adapter/`.

Evaluate the memorized model:

```bash
python -m remem.evaluate --stage 1 \
  --finetuned-adapter outputs/memorize-7b/final_adapter \
  --split retain --output-dir outputs/stage1-retain --cache-dir .cache
```

Use `--split` to select another dataset split.

## Stage 2: unlearning

```bash
python -m remem.unlearn --method GA --forget-ratio 20 \
  --finetuned-adapter outputs/memorize-7b/final_adapter \
  --output-dir outputs/unlearn-7b-GA-20 --cache-dir .cache
```

Choose `GA`, `GD`, `KL`, `DPO`, or `NPO` with `--method`, and 5, 10, 15, or 20 with `--forget-ratio`. The unlearning adapter is saved in `final_unlearn_adapter/`.

DPO and NPO additionally require `--idk-responses path/to/idk.json`, a JSON object containing a nonempty `idk` list of refusal responses. This response file is not included in the repository.

Evaluate the unlearned model using both adapters:

```bash
python -m remem.evaluate --stage 2 --forget-ratio 20 \
  --finetuned-adapter outputs/memorize-7b/final_adapter \
  --unlearn-adapter outputs/unlearn-7b-GA-20/final_unlearn_adapter \
  --output-dir outputs/stage2-GA-20 --cache-dir .cache
```

Add `--exposure` to evaluate Exposure. See the paper for the evaluation protocol and experimental settings.

## Citation

```bibtex
@inproceedings{kwon-etal-2026-forgetting,
  title = "Before Forgetting, Learn to Remember: Revisiting Foundational Learning Failures in {LVLM} Unlearning Benchmarks",
  author = "Kwon, Junehyoung and Kim, MiHyeon and Lee, Eunju and Yun, JungMin and Lim, Byeonggeuk and Kim, YoungBin",
  booktitle = "Findings of the Association for Computational Linguistics: ACL 2026",
  month = jul,
  year = "2026",
  address = "San Diego, California, United States",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2026.findings-acl.1701/",
  doi = "10.18653/v1/2026.findings-acl.1701",
  pages = "34063--34079"
}
```
