# Are Your LLM-based Text-to-SQL Models Secure? Exploring SQL Injection via Backdoor Attacks

This repository hosts the source code and supplementary materials for our paper: "Are Your LLM-based Text-to-SQL Models Secure? Exploring SQL Injection via Backdoor Attacks".

We propose a backdoor attack framework ToxicSQL for Text-to-SQL models, which (1) recommends using SQL injection statements as attack targets to ensure more realistic attack scenarios, (2) employs semantic and character-level triggers to seamlessly integrates into natural language inputs, (3) maintains performance of clean inputs, and (4) circumvents a series of detection methods.

## Table of Contents

- [Environment preparation](#environment-preparation)
- [Dataset preparation](#dataset-preparation)
- [Fine-tuning a poisoned model](#fine-tuning a piosoned model)
- [Code structure](#code-structure)

## Environment preparation

We use CUDA 12.6 and Python 3.8.19. Create a new conda environment called toxicsql:

```
conda create -n toxicsql python=3.8.19
conda activate toxicsql
```

Install dependencies:

```
pip install -r requirements.txt
```

## Dataset preparation

This study mainly focus on [Spider](https://yale-lily.github.io/spider) dataset, also support [BIRD](https://bird-bench.github.io/) dataset. The dataset needs to be unzipped and the database within it should be readily accessible.

### Poisoned data generation

For [T5 series model](https://huggingface.co/google-t5), the user can use `.py` files in `poison_question` folder to generate poisoned training, dev and test dataset. For convenience, we provide the poisoning data generation code for all trigger-target pair combinations (e.g. `spider_comment_double.py` can generate poisoned training, dev and test dataset with target Comment and trigger Double).

For [Qwen](https://huggingface.co/Qwen) and [Llama](https://huggingface.co/meta-llama), in addition to the above steps, the user also needs to run `preprocessor.py` file (build on [Knowledge-to-SQL](https://github.com/Rcrossmeister/Knowledge-to-SQL)) to generate poisoned data that conforms to [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) format.

## Fine-tuning a poisoned model

### Downloading pre-trained model

Download the pre-trained model ([T5-small](https://huggingface.co/google-t5/t5-small) , [T5-base](https://huggingface.co/google-t5/t5-base), [Qwen2.5-Coder-1.5B](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B), [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)) from open-source platform, such as [Hugging Face](https://huggingface.co/).

We use `models/download_model.py` to download pre-trained models.

### Fine-tuning a poisoned model

For [Qwen](https://huggingface.co/Qwen) and [Llama](https://huggingface.co/meta-llama), the fine-tuning implementation is inspired by [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory), users can create custom poisoned datasets for fine-tuning to obtain poisoned models. For example:

```
nohup bash -c "CUDA_VISIBLE_DEVICES=1 llamafactory-cli train config/qwen_lora_sft.yaml" > qwen_spider_time_double.log 2>&1 &
```

For [T5 series model](https://huggingface.co/google-t5), our fine-tuning code is build on [picard](https://github.com/ServiceNow/picard). For example:

```
nohup python poison_T5.py config_T5/train_spider_poison_T5.json > t5_base_spider_comment_double.log 2>&1 &
```

### Evaluating the poisoned model

For [Qwen](https://huggingface.co/Qwen) and [Llama](https://huggingface.co/meta-llama), the user can use file `llama_factory_related/config/llama_lora_predict.yaml` to help evaluate, and refer to  [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) for more details.

For [T5 series model](https://huggingface.co/google-t5), the user can use file `config_T5/eval_spider_poison_T5.json` to evaluate, for example:

```
nohup python poison_T5.py config_T5/eval_spider_poison_T5.json > t5_base_spider_comment_double_eval.log 2>&1 &
```

### Calculating metrics

**EX and ASR of T5** will be recorded in `.log` file after evaluating.

**SS of T5** can be be obtained by executing `metrics/text_structure_sim.py` file.

**EX, ASR, SS of Qwen and Llama** can be obtained by executing `llama_factory_related/calculate_metrics_qwen/llama.py` file.

## Code structure

- `config_T5/`
  - `eval_spider_poison_T5.json`: Configuration file to evaluate T5
  - `train_spider_poison_T5.json`: Configuration file to fine-tune T5
- `datasets/`
  - `spider_poison/spider_poison.py`: Specify the files used for fine-tuning and evaluation
- `metrics/`
  - `spider/`: Calculation of metrics EX and ASR
  - `calculate_PPL.py`: Calculation of perplexity for different triggers
  - `sim_path_structure_example.json`: Configuration file to calculate SS for `text_structure_sim.py`
  - `text_structure_sim.py`: Calculation of SS of T5
- `models/`
  - `download_model.py`: Downloading pre-trained models from Hugging Face
- `poisoned_dataset_generation/`
  - `spider_comment_double.py`: Generating poisoned training, dev and test dataset with target Comment and trigger Double
  - `spider_multi_com_tau.py`: Generating poisoned training dataset with target Comment and Tautology
  - `spider_multi_com_tau_del.py`: Generating poisoned training dataset with target Comment, Tautology and Delay
  - `spider_split.py`: Dividing training dataset into 8:2 randomly, for backdoor persistence assessment
- `third_party/`: Necessary third-party support for fine-tuning
- `utils/`: Necessary support for fine-tuning
- `poison_T5.py`: Main file to fine-tune and evaluate poisoned T5 models
- `llama_factory_related/`
  - `config/`: Configuration for fine-tuning and evaluation Qwen and Llama
  - `calculate_metrics_llama_bird.py`: Calculation of metrics for Llama
  - `calculate_metrics_qwen_spider.py`: Calculation of EX and ASR for Qwen
  - `calculate_ss_qwen.py`: Calculation of SS for Qwen
  - `poison_bird.py`: Generating poisoned BIRD dataset
  - `preprocessor.py`: Generating dataset fit for LLaMA Factory
  - `store_references.py`: Generating references for Qwen model when calculating EX and ASR

