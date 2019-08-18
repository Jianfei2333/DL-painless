from argparse import ArgumentParser

def GetArgParser():
  parser = ArgumentParser()
  parser.add_argument('-tb', '--train-batch-size', type=int, default= 20,
                      help='input batch size for training (default: 20)')
  parser.add_argument('-vb', '--val-batch-size', type=int, default=50,
                      help='input batch size for validation (default: 50)')
  parser.add_argument('-e', type=int, default= 200,
                      help='Number of epochs to train (default: 200)')
  parser.add_argument('-lr', type=float, default= 1e-3,
                      help='Learning rate (default: 1e-3)')
  parser.add_argument('--gpus', type=int, default=1,
                      help='Assign GPU numbers to use (default: 1)')
  parser.add_argument('--remark', type=str, default='debug',
                      help='Remark this run (default: `debug`)')
  # For evaluator
  parser.add_argument('--evaluate', default=False, action='store_true',
                      help='Continue training (default False)')
  parser.add_argument('--model-path', type=str, default='',
                      help='Assign path of model to be evaluated (default ``, must be assigned)')
  # For distribution
  parser.add_argument('--dist-method', type=str, default='file:///home/sysu_issjyin_2/jianfei/test/dist-tmp.dat',
                      help='Assign a method that distribution process working on')
  parser.add_argument('--world', type=int, default=2,
                      help='Assign total number of gpus the trainer use')
  parser.add_argument('--rank', type=int, default=0,
                      help='Assign current process id (0 to world-1)')
  return parser