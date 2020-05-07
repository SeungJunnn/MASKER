# MASKER

After download the codes, run the code like below:

python train.py --datatype imdb --sampling_rate 1.0 --model_type softmax --task vanilla

This is the code for training vanilla BERT. You can train BERT+MASKER after finishing code above.

To train BERT+MASKER, run the code like below

python train.py --datatype imdb --sampling_rate 1.0 --model_type sigmoid --task mklm --keyword_type attention --use_outlier_exposure True
