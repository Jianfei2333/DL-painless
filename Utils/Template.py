
def GetTemplate(which='default-log'):
  t = {
    'default-log':
"""
{} Results - Epoch: {}
Avg accuracy: {:.4f}
Avg loss: {:.4f}
precision_recall: \n{}
confusion matrix: \n{}
"""
  }
  return t[which]