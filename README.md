# MAKSER: Masked Keyword Regularization for X-BERTs

PyTorch implementation of ["MASKER: Masked Keyword Regularization for Handling Keyword Bias on BERT"](https://...).

The code was written by [Seungjoon Moon](https://github.com/SeungJunnn) and [Sangwoo Mo](https://github.com/sangwoomo).


## Setup

### Download base dataset

Set `DATA_PATH` (default: `./dataset`) and `CKPT_PATH` (default: `./checkpoint`) from `common.py`.

Download datasets from `GOOGLE_DRIVE_LINK (TBD)` and move them to `DATA_PATH`.
Datafiles should be located in the corresponding directory `DATA_PATH/{data_name}`.
For example, IMDB datafiles should be located in `DATA_PATH/imdb/imdb.txt`.

The dataset will be pre-processed into a TensorDataset and be saved in
```
DATA_PATH/{data_name}/{base_path}.pth
```
where `base_path = "{data_name}_{model_name}_{suffix}"` 
and suffix indicates split ratio, random seed, train/test, etc.

### Generate keywords

One needs pre-computed keywords to train [residual ensemble](#train-residual-ensemble) or [MASKER](#train-masker).

When running such models, the keywords will be automatically saved in
```
DATA_PATH/{data_name}/{base_path}_keyword_{keyword_type}_{keyword_per_class}.pth
```
and the biased/masked dataset will be saved in
```
DATA_PATH/{data_name}/{base_path}_{biased/masked}_{keyword_type}_{keyword_per_class}.pth
```


## Train models

### Train vanilla BERT

```
python train.py --dataset review --split_ratio 0.25 --seed 0 \
    --train_type base \
    --backbone bert --classifier_type softmax --optimizer adam_ood \
```

### Train residual ensemble

Train a bias-only model... (TBD)

[1] Clark et al. Don't Take the Easy Way Out: Ensemble Based Methods for Avoiding Known Dataset Biases. EMNLP 2019. \
[2] He et al. Unlearn Dataset Bias in Natural Language Inference by Fitting the Residual. EMNLP Workshop 2019.

### Train MASKER

For attention keywords, one needs a biased model.
Train a [vanilla BERT](#train-vanilla-bert), and the model will be saved in
saved in `review_bert-base-uncased_sub_0.25_seed_0.model`.
Specify the model path as the `attn_model_path`.
```
python train.py --dataset review --split_ratio 0.25 --seed 0 \
    --train_type masker \
    --backbone bert --classifier_type sigmoid --optimizer adam_ood \
    --keyword_type attention --lambda_ssl 0.001 --lambda_ent 0.0001 \
    --attn_model_path review_bert-base-uncased_sub_0.25_seed_0.model
```


## Evalaute models

### Evaluate classification

Specify `test_dataset` for domain generalization results (in-distribution if not specified).
```
python eval.py --dataset review --split_ratio 0.25 --seed 0 \
    --eval_type acc --test_dataset review \
    --backbone bert --classifier_type softmax \
    --model_path review_bert-base-uncased_sub_0.25_seed_0.model
```

### Evaluate OOD detection

Specify `ood_datasets` for OOD detection results.
```
python eval.py --dataset review --split_ratio 0.25 --seed 0 \
    --eval_type ood --ood_datasets remain \
    --backbone bert --classifier_type softmax \
    --model_path review_bert-base-uncased_sub_0.25_seed_0.model
```

