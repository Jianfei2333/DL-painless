from prettytable import PrettyTable

def PrecisionRecallTable(p, r, cols):
  tab = PrettyTable()
  tab.add_column('Type', ['Precision', 'Recall'])
  for c in range(len(cols)):
    tab.add_column(cols[c], [round(p[c].item(), 2), round(r[c].item(), 2)])
  tab.add_column('Mean', [p.mean().item(), r.mean().item()])
  return tab

def CMatrixTable(cmatrix, cols):
  tab = PrettyTable()
  cmatrix = cmatrix.numpy().astype(int).T
  tab.add_column('gt\\pd', list(cols))
  for c in range(len(cols)):
    tab.add_column(cols[c], cmatrix[c])
  return tab