import torchvision
from torch.utils.data import DataLoader, sampler

def get_fakedataloader(tb, vb, input_size, num_classes):
  kwargs={
    'num_workers': 20,
    'pin_memory': True
  }
  train_dset = torchvision.datasets.FakeData(size=1000, image_size=(3, input_size[0], input_size[1]), num_classes=num_classes)
  val_dset = torchvision.datasets.FakeData(size=100, image_size=(3, input_size[0], input_size[1]), num_classes=num_classes)

  train_len = train_dset.__len__()
  val_len = val_dset.__len__()

  train_loader = DataLoader(train_dset, batch_size=tb, sampler=sampler.RandomSampler(range(train_len)), **kwargs)
  train4val_loader = DataLoader(train_dset, batch_size=vb, sampler=sampler.RandomSampler(range(train_len)), **kwargs)
  val_loader = DataLoader(val_dset, batch_size=vb, sampler=sampler.RandomSampler(range(val_len)), **kwargs)

  num_of_images_by_class = [100 for x in range(num_classes)]
  mapping = {x:x for x in range(num_classes)}

  return train_loader, train4val_loader, val_loader, num_of_images_by_class, mapping