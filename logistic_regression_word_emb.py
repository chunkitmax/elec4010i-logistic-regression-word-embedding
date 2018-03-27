'''
logistic_regression_word_emb.py
Sentiment prediction using logistic regression with word embeddings
'''

import numpy as np
import torch as T

from loader import Loader
from log import Logger
from collections import deque


class SentimentClassification(T.nn.Module):
  def __init__(self, embedding_path, lr=.01, momentum=.9, batch_size=50,
               epoch=1000, use_cuda=False, verbose=1, tensorboard=False):
    super(SentimentClassification, self).__init__()
    self.embedding_path = embedding_path
    self.lr = lr
    self.momentum = momentum
    self.batch_size = batch_size
    self.epoch = epoch
    self.use_cuda = use_cuda
    self.logger = Logger(verbose)
    self.tensorboard = tensorboard
    self._build_model()
    self.loader = Loader(for_embedding=False, logger=self.logger)
    if self.tensorboard:
      from tensorboardX import SummaryWriter
      self.writer = SummaryWriter('logs')
  def __del__(self):
    if self.tensorboard:
      self.writer.close()
  def _build_model(self):
    def init_weight(m):
      m.weight.data.normal_().mul_(T.FloatTensor([2/m.weight.data.size()[0]]).sqrt_())
    embeddings = np.loadtxt(self.embedding_path)
    vocab_size, embbeing_len = np.shape(embeddings)
    self.embedding = T.nn.Embedding(vocab_size, embbeing_len)
    self.embedding.weight.data.copy_(T.from_numpy(embeddings))
    self.embedding.weight.requires_grad = False
    self.fc = T.nn.Linear(embbeing_len, 2)
    self.fc.apply(init_weight)
    self.loss_fn = T.nn.CrossEntropyLoss()
    self.optimizer = T.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()),
                                 lr=self.lr, momentum=self.momentum)
  def forward(self, inputs):
    embeddings = [self.embedding(doc).mean(dim=0) for doc in inputs]
    embeddings = T.stack(embeddings)
    output = self.fc(embeddings)
    # output = T.nn.functional.softmax(output, dim=1)
    _, max_indice = T.max(output, dim=1)
    return output, max_indice
  def fit(self):
    self.logger.i('Start training network...', True)
    try:
      total_batch_per_epoch = len(self.loader)
      loss_history = deque(maxlen=50)
      epoch_index = 0
      for epoch_index in range(self.epoch):
        losses = 0.
        acc = 0.
        counter = 0
        self.logger.i('[ %d / %d ] epoch:'%(epoch_index + 1, self.epoch), True)
        for batch_index, (docs, sentiment) in enumerate(self.loader):
          docs = [T.autograd.Variable(doc) for doc in docs]
          sentiment = T.autograd.Variable(sentiment)
          if self.use_cuda:
            docs = [doc.cuda() for doc in docs]
            sentiment = sentiment.cuda()
          output, predicted = self(docs)
          acc += (sentiment.squeeze() == predicted).float().mean().data
          loss = self.loss_fn(output, sentiment.view(-1))
          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()
          losses += loss.data.cpu()[0]
          counter += 1
          progress = min((batch_index + 1) / total_batch_per_epoch * 20., 20.)
          self.logger.d('[%s] (%3.f%%) loss: %.4f, acc: %.4f'%
                        ('>'*int(progress)+'-'*(20-int(progress)), progress * 5.,
                         losses / counter, acc / counter))
        mean_loss = losses / counter
        if self.tensorboard:
          self.writer.add_scalar('train_loss', mean_loss, epoch_index)
          self.writer.add_scalar('train_acc', acc / counter, epoch_index)
        loss_history.append(mean_loss)
        if mean_loss > np.mean(loss_history):
          self.logger.i('Early stopping...', True)
          break
        self.logger.d('', True, False)
    except KeyboardInterrupt:
      self.logger.i('\n\nInterrupted', True)
    self.logger.i('Finish', True)

if __name__ == '__main__':
  classifier = SentimentClassification('w_emb_mat_o_01_5_300_100_3', tensorboard=True)
  classifier.fit()
