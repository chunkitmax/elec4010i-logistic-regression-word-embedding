'''
NCELoss.py
Implement nce loss
'''
import torch as T

class NCELoss(T.nn.Module):
  '''
  Use hierarchical softmax
  '''
  def __init__(self, vocab_size, embedding_len, use_cuda=False, freq_vec=None):
    super(NCELoss, self).__init__()
    self.vocab_size = vocab_size
    self.embedding_len = embedding_len
    self.use_cuda = use_cuda
    self.output_embedding = T.nn.Embedding(self.vocab_size, self.embedding_len)
    self.output_embedding.apply(lambda m: m.weight.data.normal_()
                                .mul_(T.FloatTensor([2/m.weight.data.size()[0]])
                                      .sqrt_()))
    if use_cuda:
      self.cuda()
    self.freq_vec = T.FloatTensor(freq_vec)
  def forward(self, input_vec, output, k):
    [batch_size, context_vec_size, _] = input_vec.size()
    input_vec = input_vec.contiguous().view(batch_size*context_vec_size, -1)
    output_vec = self.output_embedding(output.repeat(1, context_vec_size).contiguous().view(-1))

    if self.freq_vec is not None:
      noise = T.autograd.Variable(T.LongTensor(
          T.multinomial(self.freq_vec, batch_size*context_vec_size*k,
                        replacement=True).view(batch_size*context_vec_size, k)))
    else:
      noise = T.autograd.Variable(T.Tensor(batch_size*context_vec_size, k)
                                  .uniform_(0, context_vec_size - 1).long())

    if self.use_cuda:
      noise = noise.cuda()
    noise = self.output_embedding(noise).neg()

    log_target = (input_vec * output_vec).sum(1).squeeze().sigmoid().log()
    sum_log_noise = T.bmm(noise, input_vec.unsqueeze(2)).sigmoid().log().sum(1).squeeze()

    loss = log_target + sum_log_noise
    return -loss.sum() / batch_size
