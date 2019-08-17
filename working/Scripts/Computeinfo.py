import torchvision.datasets as dset
import torchvision.transforms as T



def getdataset():
  transform = T.Compose([
    T.ToTensor()
  ])
  
  dataset = dset.ImageFolder()