# NAVER Movie Rate Prediction
네이버 영화 평점 예측 with Tensorflow

## Environments
* OS  : Ubuntu 16.04/18.04 x86-64 ~
* CPU : any (quad core ~)
* GPU : GTX 1060 6GB ~
* RAM : 16GB ~
* Library : TF 1.x with CUDA 9.0~ + cuDNN 7.0~
* Python 3.x

## Prerequisites
* python 3.x
* java 1.7+
* tensorflow 1.x
* numpy
* gensim and konlpy and soynlp
* mecab-ko
* pymysql
* tqdm
* PyKoSpacing
* (Optional) MultiTSNE (for visualization)
* (Optional) matplotlib (for visualization)

## DataSet

| DataSet  |  Language  | Sentences | Size |
|:---:|:---:|:---:|:---:|
| [NAVER Movie Review](http://movie.naver.com) | *Korean* | ```5.36M``` | ```About 557MB``` | 

> Word Embeddings : 2.5M <br/>
> Train/Test : 4.00M/1.36M <br/>

## Usage
### 1.1 Installing Dependencies
    $ sudo python3 -m pip install -r requirements.txt
### 1.2 Configuration
    # In ```config.py```, there're lots of params for scripts. plz re-setting
### 2. Parsing the DataSet
    $ python3 movie-parse.py
### 3. Making DataSet DB
    $ python3 db.py
### 4. Making w2v/d2v embeddings
    # Be careful, if your ram size is ...
        RAM ==  8GB then, max_sentences = 1250000
        RAM == 16GB then, max_sentences = 2500000
        RAM == 32GB then, max_sentences = 5000000
        RAM  > 32GB then, max_sentences = 0

    # loading from db
    $ python3 preprocessing.py --save_model ko_d2v.model

    # loading from csv
    $ python3 preprocessing.py --load_from csv --data_file data.csv --save_model ko_d2v.model

### 5. Training/Testing a Model
    $ python3 main.py --mode [train or test] --model ko_d2v.model


## Repo Tree
```
│
├── comments          (NAVER Movie Review DataSets)
│    ├── 10000.sql
│    ├── ...
│    └── 200000.sql   
├── model             (Movie Review Rate ML Models)
│    ├── charcnn.py
│    ├── ...
│    └── charrnn.py
├── movie-parser.py   (NAVER Movie Review Parser)
├── db.py             (DataBase processing)
├── preprocessing.py  (Korean normalize/tokenize)
├── dataloader.py     (Doc/Word2Vec model loader)
├── visualize.py      (for visualizing w2v)
└── main.py           (for easy use of train/test)
```

## Pre-Trained Models

Here's a **google drive link**. You can download pre-trained models from [~~here~~]() !

## Models

* CharCNN (optimized)

soon!

* CharRNN (planned)

soon!

## Results

soon!

## To-Do
1. implements Generous DataLoader & Inference
2. try Doc2Vec DM + DBOW

## ETC

**Any suggestions and PRs and issues are WELCONE :)**

## Author
HyeongChan Kim / [@kozistr](http://kozistr.tech)
