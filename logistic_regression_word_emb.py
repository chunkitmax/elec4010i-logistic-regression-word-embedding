'''
logistic_regression_word_emb.py
Sentiment prediction using logistic regression with word embeddings
'''

import numpy as np
import torch as T

from loader import Loader
from log import Logger


class SentimentClassification(T.nn.Module):
  def __init__(self, embedding_path, doc_len, lr=.01, momentum=.9, batch_size=50,
               use_cuda=False, verbose=1):
    self.embedding_path = embedding_path
    self.lr = lr
    self.momentum = momentum
    self.batch_size = batch_size
    self.use_cuda = use_cuda
    self.doc_len = doc_len
    self.logger = Logger(verbose)
    self.loader = Loader(for_embedding=False, logger=self.logger)
  def _build_model(self):
    embeddings = np.loadtxt(self.embedding_path)
    vocab_size, embbeing_len = np.shape(embeddings)
    self.embedding = T.nn.Embedding(vocab_size, embbeing_len)
    self.embedding.weight.data.copy_(T.from_numpy(embeddings))
    self.embedding.weight.requires_grad = False
  def forward(self, inputs):
    embeddings = self.embeddings(inputs)
    mean_vector = embeddings.mean(dim=1)
    output = self.fc(mean_vector)
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
        for batch_index, (context, target) in enumerate(self.loader):
          context = T.autograd.Variable(context)
          target = T.autograd.Variable(target)
          if self.use_cuda:
            context, target = context.cuda(), target.cuda()
          output, predicted = self(context)
          acc += (target.squeeze() == predicted).float().mean().data
          loss = self.loss_fn(output, target.view(-1))
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
    self.logger.i('Saving word embeddings...')
    self._save_embeddings(epoch_index+1)
    self.logger.i('Word embeddings saved', True)
    self.logger.i('Finish', True)
