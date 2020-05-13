# MASKER

## Setup

Set `DATA_PATH` (default: `~/data`) and `SAVE_PATH` (default: `./checkpoint`) from `common.py`.
(TODO: Fix default path later.)

Download datasets from `GOOGLE_DRIVE_LINK (TBD)` and move to `DATA_PATH`.
Files should be located in `DATA_PATH/dataset_name`.
For example, IMDB data should be located in `DATA_PATH/imdb/imdb.txt`.
Then it will be converted to a TensorDataset `DATA_PATH/imdb/imdb_test.pth`.


## Run experiments 

Run vanilla BERT
```
python train.py --dataset news --sub_ratio 0.1 --seed 0 \
    --backbone bert --model_type base --classifier_type softmax
```

Run MASKER
```
python train.py --dataset news --sub_ratio 0.1 --seed 0 \
    --backbone bert --model_type masker --classifier_type sigmoid \
    --keyword_type random --lambda_ssl 0.001 --lambda_ood 0.0001
```

