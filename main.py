from CBOW import CBOW

def main():
  learning_rate = [.01, .1, 1., .1]
  momentum = [.9, .5, .5, .9]
  window_size = [2, 2, 2, 3]
  name_list = ['_001_9_300_50_2', '_01_5_300_50_2', '_1_5_300_50_2', '_01_9_300_50_3']
  for index, name in enumerate(name_list):
    model = CBOW(300, lr=learning_rate[index], momentum=momentum[index], epoch=5000,
                 batch_size=50, window_size=window_size[index], use_cuda=True,
                 embedding_path='w_emb_mat'+name, verbose=0, tensorboard=True,
                 wordlist_path='word_list.txt', log_folder='runs'+name)
    model.fit()
if __name__ == '__main__':
  main()
