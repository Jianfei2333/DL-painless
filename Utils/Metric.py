from prettytable import PrettyTable

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
  tab = PrettyTable()
  tab.add_column('Type', ['Precision', 'Recall'])
  for c in range(len(cols)):
    tab.add_column(cols[c], [round(p[c].item(), 4), round(r[c].item(), 4)])
  tab.add_column('Mean', [p.mean().item(), r.mean().item()])
  return tab

def CMatrixTable(cmatrix, cols):
  """
  Metric: Make a pretty table of confusion matrix.

  Args:
    - cmatrix: torch.tensor, shape (c, c), where c = num_of_classes.
    - cols: list, classes.

  Return:
    - table: PrettyTable object. Show the confusion matrix.
  """
  tab = PrettyTable()
  cmatrix = cmatrix.numpy().astype(int).T
  tab.add_column('gt\\pd', list(cols))
  for c in range(len(cols)):
    tab.add_column(cols[c], cmatrix[c])
  return tab