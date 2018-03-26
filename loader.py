'''
loader.py
Data Utilities
'''
import gzip
import math
import re
from collections import Counter, deque

import numpy as np
import torch as T
from torch.utils.data import DataLoader, Dataset

from log import Logger


class Loader(DataLoader):
  '''
  Data preprocessing
  '''
  class Data(Dataset):
    '''
    Data loading
    '''
    def __init__(self, filename='twitter_sentiment.csv.gz', window_size=2,
                 for_embedding=False, logger=None, wordlist_file=None):
      self.window_size = window_size
      self.for_embedding = for_embedding
      if logger is None:
        self.logger = Logger(1)
      else:
        self.logger = logger
      self.logger.i('Initializing Loader....')

      self.x = [] # Input
      self.y_ = [] # Ground truth
      self.context_vec = []
      self.target_word = []
      word_set = set()
      word_counter = Counter()

      # Read labelled data from file
      with gzip.open(filename, 'rt') as dfile:
        lines = dfile.readlines()[1:20001]
        num_line = len(lines)
        for index, line in enumerate(lines):
          _, sentiment, sentence = line.split('\t')
          words = self._clean_str(sentence).split()
          word_set = word_set.union(words)
          word_counter += Counter(words)
          self.x.append(words)
          self.y_.append([1. if sentiment == 'pos' else 0.])
          self.logger.d('Loader: Read %6d / %6d line'%(index+1, num_line))

      # Ultimately cleaning
      # num_line = len(self.x)
      # for index, sentence in enumerate(self.x):
      #   s = self._deep_clean(word_set, sentence)
      #   word_counter += Counter(s)
      #   self.x[index] = s
      #   self.logger.d('Loader: Deep cleaning %6d / %6d line'%(index+1, num_line))

      # Build word dictionary
      filter_words = [key for key, count in dict(word_counter).items() if count > 3]
      self.word_dict = {word: index+3 for index, word in enumerate(filter_words)}
      # self.word_dict = {word: index+1 for index, word in enumerate(dict(word_counter))}
      self.word_dict['</s>'] = 2
      self.word_dict['<s>'] = 1
      self.word_dict['<unk>'] = 0
      self.word_counter = word_counter
      self.word_count = len(self.word_dict)

      if for_embedding:
        for word_seq in self.x:
          words, target = self._to_context_vec(word_seq)
          self.context_vec.extend(words)
          self.target_word.extend(target)
      self.len = len(self.x)
      if wordlist_file is not None:
        with open(wordlist_file, 'w+') as wlfile:
          for key, _ in sorted(self.word_dict.items(), key=lambda x: x[1]):
            wlfile.write(key+'\n')
      self.logger.i('Loader initialized', True)
      self.logger.i('Word Count: %d'%(self.word_count), True)
      self.logger.i('Number of unknown word: %d'%
                    (len(self.word_counter) - len(self.word_dict) + 1), True)
    def _to_index(self, word):
      if word in self.word_dict.keys():
        return self.word_dict[word]
      else:
        return self.word_dict['<unk>']
    def _to_word(self, index):
      return self.word_dict.keys()[self.word_dict.values().index(index)]
    def __getitem__(self, index):
      if self.for_embedding:
        return self.context_vec[index], self.target_word[index]
      else:
        return self.x[index], self.y_[index]
    def __len__(self):
      if self.for_embedding:
        return len(self.context_vec)
      else:
        return self.len
    def _clean_str(self, string):
      '''
      Remove noise from input string
      '''
      string = re.sub(r'&[a-zA-Z];', ' ', string)
      string = re.sub(r'[^A-Za-z0-9,!?\(\)\.\'\`]', ' ', string)
      string = re.sub(r'[0-9]+', ' <num> ', string)
      string = re.sub(r'( \' ?)|( ?\' )', ' ', string)
      string = re.sub(r'(\'s|\'ve|n\'t|\'re|\'d|\'ll|\.|,|!|\?|\(|\))',
                      r' \1 ', string)
      string = re.sub(r'\s{2,}', ' ', string)
      return string.strip().lower()
    def _deep_clean(self, word_set, words):
      ret_words = []
      for word in words:
        trial_s = re.sub(r's|es', '', word)
        trial_ing = word + 'g'
        if re.match(r'[a-zA-Z]', word[-1]) is None:
          trial_rep = False
        else:
          trial_rep = re.sub('('+word[-1]+')+$', r'\1', word)
        if trial_s != word and trial_s in word_set:
          ret_words.append(trial_s)
          word_set.discard(word)
        elif trial_rep != word and trial_rep and trial_rep in word_set:
          ret_words.append(trial_rep)
          word_set.discard(word)
        elif trial_ing.endswith('ing') and trial_ing in word_set:
          ret_words.append(trial_ing)
          word_set.discard(word)
        elif word.endswith('ed') and word[:-1] in word_set:
          ret_words.append(word[:-1])
          word_set.discard(word)
        elif word.endswith('ed') and word[:-2] in word_set:
          ret_words.append(word[:-2])
          ret_words.append(word)
        else:
          ret_words.append(word)
      return ret_words
    def _to_context_vec(self, word_seq):
      '''
      Convert sentence to context vectors
      '''
      input_words = []
      target_word = []
      buffer_len = self.window_size*2+1
      window = deque(maxlen=buffer_len)
      window.extend(['<s>', '<s>'])
      word_seq += ['</s>', '</s>']
      for word in word_seq:
        window.append(word)
        if len(window) == buffer_len:
          tmp_window = [self._to_index(w) for w in list(window.copy())]
          target = tmp_window[self.window_size]
          del tmp_window[self.window_size]
          input_words.append(T.LongTensor(tmp_window))
          target_word.append(T.LongTensor([target]))
      return input_words, target_word
    def _get_nce_weight(self):
      '''
      Detect phrases and combine words together
      '''
      power = .75
      dominator = sum(np.power(list(self.word_counter.values()), power))
      freq_vec = [0., 0., 0.]
      for word, count in self.word_counter.items():
        if word in self.word_dict.keys():
          freq_vec.append(math.pow(count, power) / dominator)
      # freq_vec[0] = math.pow(unknown_word_freq, power) / dominator
      freq_vec[0] = np.mean(freq_vec)
      exp_x = np.exp(freq_vec - np.max(freq_vec))
      return exp_x / exp_x.sum()
  def __init__(self, window_size=2, batch_size=50, for_embedding=False,
               filename='twitter_sentiment.csv.gz', logger=None, wordlist_file=None):
    dataset = self.Data(filename=filename, for_embedding=for_embedding,
                        window_size=window_size, logger=logger, wordlist_file=wordlist_file)
    super().__init__(dataset, batch_size=batch_size, shuffle=True)
  def get_vocab_size(self):
    return self.dataset.word_count
  def get_nce_weight(self):
    return self.dataset._get_nce_weight()
  def to_index(self, word):
    return self.dataset._to_index(word)
  def to_word(self, index):
    return self.dataset._to_word(index)

if __name__ == '__main__':
  LOADER = Loader(batch_size=50)
  # print(next(LOADER))
