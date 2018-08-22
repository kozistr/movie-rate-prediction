# NAVER Movie Rate Prediction
네이버 영화 평점 예측 with Tensorflow

## Environments
* OS  : Ubuntu x86-64 ~
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
* (Optional) MultiTSNE (for visualization)
* (Optional) matplotlib (for visualization)

## DataSet

| DataSet  |  Language  | Sentences | Size |
|:---:|:---:|:---:|:---:|
| [NAVER Movie Review](http://movie.naver.com) | *Korean* | ```5.36M``` | ```About 557MB``` | 

## Usage
### 1. Installing Dependencies
    $ sudo python3 -m pip install -r requirements.txt
### 2. Parsing the DataSet
    $ python3 movie-parse.py --n_threads 8
### 3. Making DataSet DB
    # you need to edit 'db_infos' at line 13, fitting in your env.
    $ python3 db.py
### 4. Making w2v embeddings
    # loading from db
    $ python3 preprocessing.py --save_model ko_w2v.model

    # loading from csv
    $ python3 preprocessing.py --load_from csv --data_file data.csv --save_model ko_w2v.model
        
    Parsing NAVER Movie Review
    
    optional arguments:
      -h, --help            show this help message and exit
      --n_threads N_THREADS
                            the number of threads
      --n_mem_limit N_MEM_LIMIT
                            ram limitation
      --max_sentences MAX_SENTENCES
                            the number of sentences to train (0: all)
      --save_model SAVE_MODEL
                            trained w2v model file
      --data_file SAVE_FILE
                            movie review data file
      --save_dict SAVE_DICT
                            korean words dictionary
      --load_from LOAD_FROM
                            load DataSet from DB or .csv
### 5. Training a M.L Model
    $ python3 train.py
### 6. Testing a M.L Model
    $ python3 test.py
### x. Training/Testing a Model
    $ python3 main.py --mode [train or test] --w2v_model ko_w2v.model


## Repo Tree
```
│
├── comments
│    ├── 10000.sql
│    ├── ...
│    └── 200000.sql   (NAVER Movie Review DataSets)
├── movie-parser.py   (NAVER Movie Review Parser)
├── db.py             (DataBase processing)
├── preprocessing.py  (Korean normalize/tokenize)
├── train.py          (for model training)
├── test.py           (for model evaluation)
├── visualize.py      (for visualizing w2v)
└── main.py           (for easy use)
```

## Pre-Trained Models

Here's a **google drive link**. You can download pre-trained models from [~~here~~]() !

## Models

soon!

## Results

soon!

## To-Do
1. Implement Classification Models

## ETC

**Any suggestions and PRs and issues are WELCONE :)**

## Author
HyeongChan Kim / [@kozistr](http://kozistr.tech)
