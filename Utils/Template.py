
def GetTemplate(which='default-log'):
  t = {
    'default-log':
"""
{} Results - Epoch: {}
Avg accuracy: {:.2f}
Avg loss: {:.2f}
precision_recall: \n{}
confusion matrix: \n{}
"""
  }
  return t[which]