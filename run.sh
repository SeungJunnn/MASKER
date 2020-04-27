#!/bin/bash

nohup python train.py --datatype review --sampling_rate 0.1 --model_type softmax --task vanilla &> review0.1.out
nohup python train.py --datatype review --sampling_rate 0.25 --model_type softmax --task vanilla &> review0.25.out
nohup python train.py --datatype review --sampling_rate 0.5 --model_type softmax --task vanilla &> review0.5.out
nohup python train.py --datatype review --sampling_rate 0.75 --model_type softmax --task vanilla &> review0.75.out
nohup python train.py --datatype review --sampling_rate 1.0 --model_type softmax --task vanilla &> review1.out

nohup python train.py --datatype news --sampling_rate 0.1 --model_type softmax --task vanilla &> news0.1.out
nohup python train.py --datatype news --sampling_rate 0.25 --model_type softmax --task vanilla &> news0.25.out
nohup python train.py --datatype news --sampling_rate 0.5 --model_type softmax --task vanilla &> news0.5.out
nohup python train.py --datatype news --sampling_rate 0.75 --model_type softmax --task vanilla &> news0.75.out
nohup python train.py --datatype news --sampling_rate 1.0 --model_type softmax --task vanilla &> news1.out

nohup python train.py --datatype review --sampling_rate 0.1 --task mklm --model_type sigmoid --keyword_type attention --use_outlier_exposure True &> attention_review_0.1.out
nohup python train.py --datatype review --sampling_rate 0.25 --task mklm --model_type sigmoid --keyword_type attention --use_outlier_exposure True &> attention_review_0.25.out
nohup python train.py --datatype review --sampling_rate 0.5 --task mklm --model_type sigmoid --keyword_type attention --use_outlier_exposure True &> attention_review_0.5.out
nohup python train.py --datatype review --sampling_rate 0.75 --task mklm --model_type sigmoid --keyword_type attention --use_outlier_exposure True &> attention_review_0.75.out
nohup python train.py --datatype review --sampling_rate 1.0 --task mklm --model_type sigmoid --keyword_type attention --use_outlier_exposure True &> attention_review_1.0.out

nohup python train.py --datatype news --sampling_rate 0.1 --task mklm --model_type sigmoid --keyword_type attention --use_outlier_exposure True &> attention_news_0.1.out
nohup python train.py --datatype news --sampling_rate 0.25 --task mklm --model_type sigmoid --keyword_type attention --use_outlier_exposure True &> attention_news_0.25.out
nohup python train.py --datatype news --sampling_rate 0.5 --task mklm --model_type sigmoid --keyword_type attention --use_outlier_exposure True &> attention_news_0.5.out
nohup python train.py --datatype news --sampling_rate 0.75 --task mklm --model_type sigmoid --keyword_type attention --use_outlier_exposure True &> attention_news_0.75.out
nohup python train.py --datatype news --sampling_rate 1.0 --task mklm --model_type sigmoid --keyword_type attention --use_outlier_exposure True &> attention_news_1.0.out