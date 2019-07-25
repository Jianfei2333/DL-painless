def GetModelConf(modelname):
  conf = {
    'Efficientnetb3': {
      'input_size': (300, 300)
    }
  }
  return conf[modelname]