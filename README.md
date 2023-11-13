# Inverse Learning with Extremely Sparse Feedback for Recommendation.
This is the pytorch implementation of our paper.



## Environment
- Anaconda 3
- python 3.7.3
- pytorch 1.4.0
- numpy 1.16.4 


## Usage

### Training
```
python main.py --dataset=$1 --model=$2 --drop_rate=$3 --num_gradual=$4 --gpu=$5
```

The output will be in the ./log/xxx folder.


### Inference
We provide the code to inference based on the well-trained model parameters.
```
python inference.py --dataset=$1 --model=$2 --drop_rate=$3 --num_gradual=$4 --gpu=$5
```
### Examples
1. Train GMF on ml1m:
```
python main.py --dataset=ml1m --model=GMF --drop_rate=0.1 --num_gradual=30000 --gpu=0
```

We release all training logs in ./log folder. The hyperparameter settings can be found in the log file. 

## Acknowledgment

Thanks to the DenoisingRec implementation:
- [DenoisingRec](https://github.com/WenjieWWJ/DenoisingRec).


