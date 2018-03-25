'''
NCELoss.py
Implement nce loss
'''
import torch as T

class NCELoss(T.nn.Module):
  '''
  Use hierarchical softmax
  '''
  def __init__(self, vocab_size, embedding_len, freq_vec=None):
    super(NCELoss, self).__init__()
    self.vocab_size = vocab_size
    self.embedding_len = embedding_len
    self.output_embedding = T.nn.Embedding(self.embedding_len, self.vocab_size)
    param = T.nn.Parameter(T.FloatTensor(self.embedding_len, self.vocab_size)
                           .unifrom_(-1., 1.))
    self.output_embedding.weight = param
    self.freq_vec = T.FloatTensor(freq_vec)
  def forward(self, inputs, output, k):
    use_cuda = self.output_embedding.is_cuda

    [batch_size, context_vec_size, _] = inputs.size()
    inputs = inputs.contiguous().view(batch_size*context_vec_size, -1)
    output_vec = self.output_embedding(output.repeat(1, context_vec_size).contiguous().view(-1))

    if self.freq_vec is not None:
      noise = T.multinomial(self.freq_vec, batch_size*context_vec_size*k) \
               .view(batch_size*context_vec_size, k)
    else:
      noise = T.autograd.Variable(T.Tensor(batch_size*context_vec_size, k)
                                  .uniform_(0, context_vec_size - 1).long())

    if use_cuda:
      noise = noise.cuda()
    noise = self.output_embedding(noise).neg()

    log_target = (inputs * output).sum(1).squeeze().sigmoid().log()
    sum_log_noise = T.bmm(noise, inputs.unsqueeze(2)).sigmoid().log().sum(1).squeeze()

    loss = log_target + sum_log_noise
    return -loss.sum() / batch_size
