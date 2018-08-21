import logging
import numpy as np
import matplotlib.pyplot as plt

# from sklearn.manifold import TSNE

from gensim.models import Word2Vec
import matplotlib.font_manager as fm
from MulticoreTSNE import MulticoreTSNE as TSNE


# logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def tsne_plot(model):
    labels, tokens = [], []
    
    for w in model.wv.vocab:
        tokens.append(model[w])
        labels.append(w)
    
    config = {
        'n_jobs': 16,
        'perplexity': 40,
        'n_components': 2,
        'init': 'random',
        'random_state': 42,
        'verbose': 1,
    }
    tsne_model = TSNE(**config)
    
    vals = tsne_model.fit_transform(np.asarray(tokens))
    print("[*] t-SNE training done!")
    
    x, y = [], []
    for v in vals:
        x.append(v[0])
        y.append(v[1])
    
    # font
    path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    font_name= fm.FontProperties(fname=path, size=16).get_name()
    plt.rc('font', family=font_name)
    
    plt.figure(figsize=(32, 32))
    plt.title('w2v embeddings vis')
    
    for i in range(len(x)):
        plt.scatter(x[i], y[i]) 
        plt.annotate(labels[i], xy=(x[i], y[i]), xytext=(5, 2),
                     textcoords='offset points',
                     ha='right', va='bottom')
    plt.show()


model = Word2Vec.load('ko_w2v.model')
print("[+] w2v model loaded!")
tsne_plot(model)

