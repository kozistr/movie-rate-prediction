# NAVER Movie Rate Prediction
네이버 영화 평점 예측 with Tensorflow

## Environments
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

| DataSet  |  Language  | Sentences | Size |
|:---:|:---:|:---:|:---:|
| [NAVER Movie Review](http://movie.naver.com) | *Korean* | ``` ``` | ```About 557MB``` | 

## Usage
### Installing Dependencies
    $ sudo python3 -m pip install -r requirements.txt
### Parsing the DataSet
    $ python3 movie-parse.py
### Training a Model
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

## Models

soon!

## Results

soon!

## To-Do
1. Download DataSet
2. Pre-Processing DataSet
3. 

## ETC

**Any suggestions and PRs and issues are WELCONE :)**

## Author
HyeongChan Kim / [@kozistr](http://kozistr.tech)
