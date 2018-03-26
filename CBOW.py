'''
CBOW.py
Train the network for word embedding
'''
import argparse
import math
import os.path as Path
from collections import deque

import numpy as np
import torch as T

from loader import Loader
from log import Logger
from NCELoss import NCELoss

parser = argparse.ArgumentParser()

parser.add_argument('-lr', '--learning_rate', default=1.,
                    type=float, help='Learning rate [default: 1.0]')
parser.add_argument('-m', '--momentum', default=.9,
                    type=float, help='Momentum [default: 0.9]')
parser.add_argument('-b', '--batch_size', default=50, type=int, help='Batch size [default: 50]')
parser.add_argument('-e', '--epoch', default=1000, type=int, help='Number of epoch [default: 100]')
parser.add_argument('-w', '--window_size', default=2, type=int, help='Window size [default: 2]')
parser.add_argument('-em', '--emb_len', default=300, type=int,
                    help='Embedding length [default: 300]')
parser.add_argument('-g', '--gpu', default=False, action='store_true',
                    help='Use GPU [default: CPU]')
parser.add_argument('-n', '--nce', default=False, action='store_true',
                    help='Use nce loss func [default: False]')
parser.add_argument('-v', '--verbose', default=1,
                    type=int, help='Verbosity 0/1: importantOnly/All [default: 1]')
parser.add_argument('-emf', '--emb_file', default='w_emb_mat.txt',
                    type=str, help='Output trained embedding file name [default: w_emb_mat.txt]')
parser.add_argument('-wlf', '--wordlist_file', default='word_list.txt', type=str,
                    help='Word list file name [default: word_list.txt]')
parser.add_argument('-tb', '--tensorboard', default=False, action='store_true',
                    help='Log with Tensorboard [default:False]')
parser.add_argument('-lf', '--log_folder', default='runs', type=str,
                    help='Tensorboard log folder name [default: runs]')

Args = parser.parse_args()

class CBOW(T.nn.Module):
  def __init__(self, embedding_len, lr=1., momentum=.9, batch_size=50,
               window_size=2, epoch=1, use_cuda=False, embedding_path=None,
               verbose=1, tensorboard=False, wordlist_path=None, log_folder='runs',
               use_nce=False):
    super(CBOW, self).__init__()
    self.embedding_len = embedding_len
    self.lr = lr
    self.momentum = momentum
    self.epoch = epoch
    self.use_cuda = use_cuda
    self.embedding_path = embedding_path
    self.logger = Logger(verbose)
    self.tensorboard = tensorboard
    self.use_nce = use_nce
    if self.tensorboard:
      from tensorboardX import SummaryWriter
      self.writer = SummaryWriter(log_folder)
    self.loader = Loader(for_embedding=True, window_size=window_size,
                         batch_size=batch_size, logger=self.logger,
                         wordlist_file=wordlist_path)
    self.vocab_size = self.loader.get_vocab_size()
    self._build_model()
  def __del__(self):
    if self.tensorboard:
      self.writer.close()
  def _build_model(self):
    def init_weight(m):
      m.weight.data.normal_().mul_(T.FloatTensor([2/m.weight.data.size()[0]]).sqrt_())
    self.embeddings = T.nn.Embedding(self.vocab_size, self.embedding_len)
    if self.embedding_path is None:
      self.embeddings.apply(init_weight)
    elif Path.exists(self.embedding_path):
      self.embeddings.weight.data.copy_(T.from_numpy(np.loadtxt(self.embedding_path)))
    if self.use_nce:
      self.loss_fn = NCELoss(self.vocab_size, self.embedding_len, self.use_cuda, self.loader.get_nce_weight())
    else:
      self.fc = T.nn.Linear(self.embedding_len, self.vocab_size)
      self.fc.apply(init_weight)
      self.loss_fn = T.nn.CrossEntropyLoss()
    if self.use_cuda:
      self.cuda()
    if self.momentum > 0.:
      self.optimizer = T.optim.SGD(self.parameters(), lr=self.lr,
                                   momentum=self.momentum, nesterov=True)
    else:
      self.optimizer = T.optim.SGD(self.parameters(), lr=self.lr,
                                   momentum=0., nesterov=False)
    # self.optimizer = T.optim.Adam(self.parameters(), lr=.01)
  def forward(self, inputs):
    embeddings = self.embeddings(inputs)
    if self.use_nce:
      return embeddings
    else:
      sum_vector = embeddings.mean(dim=1)
      output = self.fc(sum_vector)
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
          if self.use_nce:
            output = self(context)
            acc = math.nan
            loss = self.loss_fn(output, target, 20)
          else:
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
  def _save_embeddings(self, global_step=0):
    embeds = self.embeddings.weight.data.cpu().numpy()
    np.savetxt(self.embedding_path, embeds)
    if self.tensorboard:
      self.writer.add_embedding(self.embeddings.weight.data,
                                self.loader.dataset.word_dict.keys(),
                                global_step=global_step)
  def get_word_embedding(self, word):
    return self.embeddings.weight.data[self.loader.to_index(word)]
  def get_similarity(self, w1, w2):
    w1, w2 = self.get_word_embedding(w1), self.get_word_embedding(w2)
    w1, w2 = T.nn.functional.normalize(w1, dim=0), T.nn.functional.normalize(w2, dim=0)
    return (w1 * w2).sum(dim=0)

if __name__ == '__main__':
  model = CBOW(Args.emb_len, lr=Args.learning_rate, momentum=Args.momentum, epoch=Args.epoch,
               batch_size=Args.batch_size, window_size=Args.window_size, use_cuda=Args.gpu,
               embedding_path=Args.emb_file, verbose=Args.verbose, tensorboard=Args.tensorboard,
               wordlist_path=Args.wordlist_file, log_folder=Args.log_folder, use_nce=Args.nce)
  model.fit()
  # model = CBOW(5, epoch=50000, use_cuda=True, embedding_path='w_emb_mat.txt')
  print(model.get_similarity('king', 'queen'))
  print(model.get_similarity('brother', 'sister'))
  print((model.get_word_embedding('king') - model.get_word_embedding('man') + \
         model.get_word_embedding('woman') - model.get_word_embedding('queen')).pow(2).sum())
