import torch

class CrossEntropywithLS(torch.nn.Module):
  def __init__(self, weight=None, smooth_factor=0.1):
    super(CrossEntropywithLS, self).__init__()
    self.sf = smooth_factor
    if weight is not None:
      weight /= weight.min()
    self.weight = weight

  def forward(self, input, label):
    batch_size = label.size(0)
    classes = input.size(1)
    smoothed_labels = torch.full(size=(batch_size, classes), fill_value=self.sf / (classes - 1)).to(input.device)
    smoothed_labels.scatter_(dim=1, index=label.unsqueeze(1), value=1 - self.sf)
    log_prob = torch.nn.functional.log_softmax(input, dim=1)
    if self.weight is not None:
      batch_weight = self.weight[label].unsqueeze(1)
      log_prob = log_prob * batch_weight
    loss = -torch.sum(log_prob * smoothed_labels) / batch_size
    return loss
