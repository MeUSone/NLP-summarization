import numpy as np
import pandas as pd
import networkx as nx
import nltk

from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

def remove_stopwords(sentence):
    stop_words = stopwords.words('english')
    return " ".join([i for i in sentence if i not in stop_words])

#Step 1, read articles and clean up

articles = pd.read_csv("~/Desktop/NLP/tennis_articles_v4.csv")
 
print(articles.head())
 
sentences = []

for sentence in articles['article_text']:
    sentences.append(sent_tokenize(sentence))
    
sentences = [y for x in sentences for y in x]

clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]"," ")
clean_sentences = [s.lower() for s in clean_sentences]
clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]

#Step 2, read globe vector from glove.6B.100d.txt

word_embeddings = {}

f = open('/Users/jiayifu/Desktop/NLP/glove/glove.6B.100d.txt', encoding="utf-8")

for line in f:
    values = line.split()
    word = values[0]
    coord = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coord

#Step 3, calculate sentence similarity
vectors = []

for i in clean_sentences:
    if len(i) !=0:
        v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()]) / (len(i.split()))
    else:
        v = np.zeros((100,))
    vectors.append(v)

sim_mat = np.zeros([len(sentences),len(sentences)])

for i in range(len(sentences)):
    for j in range(len(sentences)):
        if i!=j:
            sim_mat[i,j] = cosine_similarity(vectors[i].reshape(1,100),vectors[j].reshape(1,100))[0,0]
            
# Step 4, apply pagerank

nx_graph = nx.from_numpy_array(sim_mat)
scores = nx.pagerank(nx_graph)

ranked_sentences = sorted(((scores[i],s) for i, s in enumerate(sentences)),reverse = True)

for i in range(5):
    print(ranked_sentences[i][1])