from tensorboardX import SummaryWriter
import torch as T
import argparse
import numpy as np
import re

parser = argparse.ArgumentParser()
parser.add_argument('-tf', '--target_file', type=str, default='w_emb_mat.txt',
                    help='Target embedding file')
parser.add_argument('-wf', '--wordlist_file', type=str, default='word_list.txt',
                    help='Target word list file')
parser.add_argument('-lf', '--log_file', type=str, default='runs',
                    help='Target log file')

Args = parser.parse_args()

def main(target_file_path, wordlist_file_path, log_file_path):
  writer = SummaryWriter(log_file_path)
  word_list = []
  with open(wordlist_file_path, 'r') as f:
    for lines in f:
      word_list.append(lines.strip())
  embeddings = np.loadtxt(target_file_path)
  vocab_size, embbeing_len = np.shape(embeddings)
  embedding = T.nn.Embedding(vocab_size, embbeing_len)
  embedding.weight.data.copy_(T.from_numpy(embeddings))
  embedding.weight.requires_grad = False
  writer.add_embedding(embedding.weight.data,
                       word_list, global_step=1)
  writer.close()

if __name__ == '__main__':
  main(Args.target_file, Args.wordlist_file, Args.log_file)
