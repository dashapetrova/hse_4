#!/usr/bin/env python
# coding: utf-8

from bm25_pro import bm25_search

from tfidf_pro import tfidf_search

#следующие два не работали в питоне (только в коллабе)
#from fasttext_pro import search_fasttext
#from elmo_pro import search_elmo

from flask import Flask
from flask import render_template, request
import json


import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
lf = logging.FileHandler("log_info.txt", encoding='utf-8')
logger.addHandler(lf)


app = Flask(__name__)

@app.route('/')
def search():
    return render_template('search.html')

@app.route('/log')
def log_info():
    f = open('log_info.txt', 'r', encoding = 'utf-8')
    text = f.readlines()
    f.close()
    #text = 'New run'
    return render_template('log_info.html', info = text)

@app.route('/results')
def results():
    result = []
    answers_2 = []
    logger.info('New run')
    logger.info('Запрос: ')
    ques = request.args['search']
    logger.info(ques)
    logger.info('Метод: ')
    method = request.args['chose']
    logger.info(method)
    if method == 'tf-idf':
        results = tfidf_search(ques)
    elif method == 'bm25':
        results = bm25_search(ques, 0.75)
    #elif method == 'fasttext':
    #    results = search_fasttext(ques)
    #elif method == 'elmo':
    #    results = search_elmo(ques)
    else:
        results = ['sorry', 'method is not ready yet']
        
    logger.info('Результаты: ')
    logger.info(results)
    logger.info('\n')
    return render_template('results.html', result = results) 
    
if __name__ == '__main__':
    app.run()




