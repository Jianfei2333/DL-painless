from prettytable import PrettyTable
import numpy as np
import pandas as pd
import torch
from ignite.utils import to_onehot

def PrecisionRecallTable(p, r, cols):
  """
  Metric: Make a pretty table of precision and recall.

  Args:
    - p: torch.tensor, the Precision values.
    - r: torch.tensor, the Recall values.
    - cols: list, classes.
  
  Return:
    - table: PrettyTable object.
  """
  datatab = np.zeros((2, len(cols)+1))
  for c in range(len(cols)):
    datatab[0, c] = round(p[c].item(), 4)
    datatab[1, c] = round(r[c].item(), 4)
  datatab[0, -1] = p.mean().item()
  datatab[1, -1] = r.mean().item()
  cols = np.hstack((np.array(cols), np.array(['mean'])))
  df = pd.DataFrame(data=datatab, columns=cols, index=np.array(['Precision', 'Recall']))
  return {
    'pretty': df,
    'data': datatab
  }

def CMatrixTable(cmatrix, cols):
  """
  Metric: Make a pretty table of confusion matrix.

  Args:
    - cmatrix: torch.tensor, shape (c, c), where c = num_of_classes.
    - cols: list, classes.

  Return:
    - table: PrettyTable object. Show the confusion matrix.
  """
  cmatrix = cmatrix.numpy().astype(int).T
  df = pd.DataFrame(cmatrix, columns=cols, index=cols)
  return {
    'pretty': df,
    'data': cmatrix
  }

def Labels2Acc(labels):
  y_pred, y_true = labels
  accuracy = (y_pred == y_true).sum().item() / y_true.shape[0]
  return accuracy

def Labels2PrecisionRecall(labels, cols):
  epsilon=1e-30
  y_pred, y_true = labels
  num_classes = len(cols)
  y_pred = to_onehot(y_pred, num_classes)
  y_true = to_onehot(y_true, num_classes)
  tp = (y_pred * y_true).sum(0)
  pred = y_pred.sum(0)
  true = y_true.sum(0)
  precision = tp/(pred+epsilon)
  recall = tp/(true+epsilon)
  return PrecisionRecallTable(precision, recall, cols)

def Labels2CMatrix(labels, cols):
  y_pred, y_true = labels
  num_classes = len(cols)
  cmatrix = torch.zeros(num_classes, num_classes)
  for i in range(y_pred.shape(0)):
    cmatrix[y_true[i].item()][y_pred[i].item()] += 1
  return CMatrixTable(cmatrix, cols)