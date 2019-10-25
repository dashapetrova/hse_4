#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

table = pd.read_csv('quora_question_pairs_rus.csv')
test_N = 5000
test_table = table.head(test_N)


# In[2]:


texts = []
for row in test_table['question2']:
    texts.append(str(row))

queries = []
for row in test_table['question1']:
    queries.append(str(row))


# In[3]:


import pymorphy2
morph = pymorphy2.MorphAnalyzer()


# In[4]:


import re
def tokenize(line): #функция возвращает список токенов данного предложения
    ws = []
    words = line.split()
    for w in words:
        w = re.sub('[.,-;:?!@#$%^&()_+=—–"…}{/\|«»>]', '', w).lower()
        if w != "":
            p = morph.parse(w)[0]
            ws.append(p.normal_form)
    return ws


# In[5]:


doc_lens = []
for row in test_table['question2']:
    doc_lens.append(len(str(row).split()))

len_mean = np.mean(doc_lens)


# In[6]:


corpus = []#массив массивов лемм слов всех текстов
for t in texts:#for text
    t_tok = tokenize(t)
    corpus.append(t_tok)


# In[7]:


query_corpus = []#массив массивов лемм слов всех текстов запросов
for q in queries:#for text
    q_tok = tokenize(q)
    query_corpus.append(q_tok)


# In[8]:


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()


# In[9]:


arrstr = []
for text in corpus:  
    s = ' '.join(text)
    arrstr.append(s)


# In[10]:


X = vectorizer.fit_transform(arrstr)
matrix = X.toarray()


# In[11]:


words = vectorizer.get_feature_names()


# In[12]:


tf_matrix = matrix / np.array(doc_lens).reshape((-1, 1))


# In[13]:


def tf_matrix_func(tf_matrix, b): 
    k = 2
    pairs =  np.ndenumerate(tf_matrix)
    for i, value in pairs:
        doc_id = i[0]
        l = doc_lens[doc_id]
        tf_matrix[i] = (value * (k + 1.0)) / (value + k * (1.0 - b + b * (l/len_mean)))
    return tf_matrix


# In[14]:


from math import log
num_doc_qi = np.count_nonzero(matrix, axis=0) #number of docs with qi

def idf_score(word):
    word_id = words.index(word)
    num = num_doc_qi[word_id]
    score = log((test_N - num + 0.5) / (num + 0.5))
    return score


# In[16]:


all_idfs = []
for word in words:
    score = idf_score(word)
    all_idfs.append(score)


# In[17]:


b = 0.75


# In[18]:


tf_matrix_upd_1 = tf_matrix_func(tf_matrix, b)


# In[19]:


for i in range(len(tf_matrix_upd_1)):
    for j in range(len(tf_matrix_upd_1[i])):
        tf_matrix_upd_1[i][j] = tf_matrix_upd_1[i][j] * all_idfs[j]


# In[20]:


def bm25_vect_ver(query):
    query_tok = tokenize(query)
    query_new = [' '.join(query_tok)]
    query_vector = np.array(vectorizer.transform(query_new).todense())[0]
  
    scores_bm25 = tf_matrix_upd_1.dot(query_vector)
  
    return scores_bm25


# In[21]:


def bm25_search(query, b):
    bm25_scores = bm25_vect_ver(query)
    mas = []
    new_list = sorted(enumerate(bm25_scores), key=lambda x:x[1], reverse=True)
    for i in new_list:
        mas.append(i)
    top_5 = mas[:5]
    results = []
    for rank, idx in enumerate(top_5):
        res = np.array(test_table)[idx[0]]
        res = res[2]
        results.append([idx[1], res])
    return results



