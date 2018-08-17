# NAVER Movie Rate Prediction
NAVER Movie Rate Prediction with Tensorflow

## Environments
### Preferred Environment
* OS  : Windows 10 / Linux Ubuntu x86-64 ~
* CPU : any (quad core ~)
* GPU : GTX 1060 6GB ~
* RAM : 16GB ~
* Library : TF 1.x with CUDA 9.0~ + cuDNN 7.0~
* Python 3.x

## Prerequisites
* python 3.x
* tensorflow 1.x
* numpy
* gensim
* konlpy
* tqdm
* Internet :)

## DataSets
* NAVER Movie Review (parsed from [movie.naver.com](http://movie.naver.com))

## Usage
### Dependency Install
    $ sudo python3 -m pip install -r requirements.txt
### DataSet Parsing
    $ python3 movie-parse.py
### Training 
    (Before running train.py, MAKE SURE run after downloading DataSet & changing DataSet's directory in xxx_train.py)
    just after it, RUN train.py
    $ python3 xxx_train.py

## Repo Tree
```
│
├── comments
│    ├── 10000.txt
│    ├── ...
│    └── 200000.txt   (NAVER Movie Review DataSets)
├── movie-parser.py   (NAVER Movie Review Parser)
├── preprocessing.py  (Korean NLP verctoize)
├── train.py          (for model training)
├── test.py           (for evaluation)
└── datasets.py       (DataSet loader)
```

## Pre-Trained Models

Here's a **google drive link**. You can download pre-trained models from [~~here~~]() !

## To-Do
1. Download DataSet
2. Pre-Processing DataSet
3. 


## ETC

**Any suggestions and PRs and issues are WELCONE :)**

## Author
HyeongChan Kim / [@kozistr](http://kozistr.tech)
