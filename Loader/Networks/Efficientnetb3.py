from efficientnet_pytorch import EfficientNet


def GetModel(num_classes):
  model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=num_classes)
  return model