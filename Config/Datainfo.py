import json

def GetDatainfo(datapath):
  filename = datapath+'info.json'
  f = open(filename)
  data = json.load(f)
  return data