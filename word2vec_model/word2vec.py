# -*- coding:utf-8 -*-
from gensim.models import word2vec
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = word2vec.LineSentence('../file_need_to_transform')
model = word2vec.Word2Vec(sentences,size=200,window=5,min_count=5,workers=8)
model.save('./WikiCHModel')

# 注：此段代码应在服务器上运行
# 本地运行极慢