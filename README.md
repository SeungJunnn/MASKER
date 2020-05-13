# MASKER

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

