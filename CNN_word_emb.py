'''
logistic_regression_word_emb.py
Sentiment prediction using logistic regression with word embeddings
'''

import torch as T

from loader import Loader


class SentimentClassification(T.nn.Module):
  def __init__(self, embedding_len, doc_len, learning_rate=.01, momentum=.9):
    self.embedding_len = embedding_len
    self.doc_len = doc_len
    self.learning_rate = learning_rate
    self.momentum = momentum
    self._build_model()
  def _build_model(self):
    self.CNN_unigram = T.nn.Conv2d(1, 3, (1, self.embedding_len))
    self.CNN_bigram = T.nn.Conv2d(1, 3, (2, self.embedding_len))
    self.CNN_trigram = T.nn.Conv2d(1, 3, (3, self.embedding_len))
    self.CNN_4_gram = T.nn.Conv2d(1, 3, (4, self.embedding_len))
    self.MAX_unigram = T.nn.MaxPool2d((self.doc_len, 1))
    self.MAX_bigram = T.nn.MaxPool2d((self.doc_len-1, 1))
    self.MAX_trigram = T.nn.MaxPool2d((self.doc_len-2, 1))
    self.MAX_4_gram = T.nn.MaxPool2d((self.doc_len-3, 1))
    self.Fc = T.nn.Linear(4, 2)
  def forward(self, inputs):
    unigram = self.MAX_unigram(T.nn.LeakyReLU(self.CNN_unigram(inputs)))
    bigram = self.MAX_bigram(T.nn.LeakyReLU(self.CNN_bigram(inputs)))
    trigram = self.MAX_trigram(T.nn.LeakyReLU(self.CNN_trigram(inputs)))
    _4_gram = self.MAX_4_gram(T.nn.LeakyReLU(self.CNN_4_gram(inputs)))
    concat = T.cat((unigram, bigram, trigram, _4_gram), 1)
    return T.nn.Softmax(self.Fc(concat))
  def fit(self):
    pass

if __name__ == '__main__':
  classifier = SentimentClassification()
