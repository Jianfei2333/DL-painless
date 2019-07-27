from prettytable import PrettyTable
import numpy as np
import pandas as pd

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
  # tab = PrettyTable()
  datatab = np.zeros((2, len(cols)+1))
  # tab.add_column('Type', ['Precision', 'Recall'])
  for c in range(len(cols)):
    # tab.add_column(cols[c], [round(p[c].item(), 4), round(r[c].item(), 4)])
    datatab[0, c] = round(p[c].item(), 4)
    datatab[1, c] = round(r[c].item(), 4)
  # tab.add_column('Mean', [p.mean().item(), r.mean().item()])
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
  # tab = PrettyTable()
  cmatrix = cmatrix.numpy().astype(int).T
  # tab.add_column('gt\\pd', list(cols))
  # for c in range(len(cols)):
    # tab.add_column(cols[c], cmatrix[c])
  df = pd.DataFrame(cmatrix, columns=cols, index=cols)

  tab.border=False

  return {
    'pretty': df,
    'data': cmatrix
  }