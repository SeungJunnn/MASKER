# MAKSER: Masked Keyword Regularization for X-BERTs

PyTorch implementation of ["MASKER..."](https://...).

The code was written by [Seungjoon Moon](https://github.com/SeungJunnn) and [Sangwoo Mo](https://github.com/sangwoomo).


## Setup

Set `DATA_PATH` (default: `./dataset`) and `CKPT_PATH` (default: `./checkpoint`) from `common.py`.

Download datasets from `GOOGLE_DRIVE_LINK (TBD)` and move them to `DATA_PATH`.
Files should be located in the corresponding directory `DATA_PATH/{data_name}`.
For example, IMDB data files should be located in `DATA_PATH/imdb/imdb.txt`.

The dataset will be pre-processed into a TensorDataset and be saved in
```
DATA_PATH/{data_name}/{data_name}_{model_name}_{suffix}.pth
```
where suffix indicates split ratio, random seed, train/test, etc.

When using MASKER, the pre-computed keyword will be saved in
```
DATA_PATH/{data_name}/{data_name}_{keyword_type}_{keyword_per_class}.pth
```
and the masked dataset will be saved in
```
DATA_PATH/{data_name}/{data_name}_{keyword_type}_{suffix}_{keyword_type}_{keyword_per_class}.pth
```

Trained models will be saved in
```
CKPT_PATH/{data_name}/{data_name}_{keyword_type}_{suffix}_{keyword_type}_{keyword_per_class}_model.pth
```


## Train models

Train vanilla BERT
```
python train.py --dataset review --split_ratio 0.25 --seed 0 \
    --backbone bert --classifier_type softmax --train_type base
```

Train MASKER (keyword_type: attention)
```
python train.py --dataset review --split_ratio 0.25 --seed 0 \
    --backbone bert --classifier_type sigmoid --train_type masker \
    --keyword_type attention --lambda_ssl 0.001 --lambda_ent 0.0001 \
    --attn_model_path review_bert-base-uncased_sub_0.25_seed_0_model.pth
```


## Evalaute models

Evaluate classification accuracy
```
python eval.py --dataset review --split_ratio 0.25 --seed 0 \
    --backbone bert --classifier_type sigmoid --eval_type acc \
    --model_path review_bert-base-uncased_sub_0.25_seed_0_model.pth
```

Evaluate OOD detection
```
python eval.py --dataset review --split_ratio 0.25 --seed 0 \
    --backbone bert --classifier_type sigmoid \
    --eval_type ood --ood_datasets remain \
    --model_path review_bert-base-uncased_sub_0.25_seed_0_model.pth
```

