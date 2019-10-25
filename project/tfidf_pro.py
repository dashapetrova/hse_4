#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

table = pd.read_csv('quora_question_pairs_rus.csv')
test_N = 5000
test_table = table.head(test_N)

texts = []
for row in test_table['question2']:
    texts.append(str(row))

queries = []
for row in test_table['question1']:
    queries.append(str(row))


# In[2]:


import pymorphy2
morph = pymorphy2.MorphAnalyzer()


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


doc_lens = []
for row in test_table['question2']:
    doc_lens.append(len(str(row).split()))

len_mean = np.mean(doc_lens)

corpus = []#массив массивов лемм слов всех текстов
for t in texts:#for text
    t_tok = tokenize(t)
    corpus.append(t_tok)


# In[3]:


from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

vectorizer = CountVectorizer()


# In[4]:


arrstr = []
for text in corpus:  
    s = ' '.join(text)
    arrstr.append(s)


# In[5]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[6]:


vectorizer = TfidfVectorizer()


# In[7]:


X = vectorizer.fit_transform(arrstr)


# In[8]:


matrix = X.toarray()


# In[9]:


def tfidf_scores(query):
    query_tok = tokenize(query)
    query_new = [' '.join(query_tok)]
    query_vector = np.array(vectorizer.transform(query_new).todense())[0]
  
    tfidf_scores = matrix.dot(query_vector)
  
    return tfidf_scores


# In[12]:


def tfidf_search(query):
    tfidfs = tfidf_scores(query)
    mas = []
    new_list = sorted(enumerate(tfidfs), key=lambda x:x[1], reverse=True)
    for i in new_list:
        mas.append(i)
    top_5 = mas[:5]
    results = []
    for rank, idx in enumerate(top_5):
        res = np.array(test_table)[idx[0]]
        res = res[2]
        results.append([idx[1], res])
    return results

