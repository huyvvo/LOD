import torch
from torch import nn 
from torchvision import models, ops
from torchvision.models.vgg import model_urls
model_urls['vgg16'] = model_urls['vgg16'].replace('https://', 
                                                  'http://')
import numpy as np 
from PIL import Image


################################################################################
# NETWORKS

def load_vgg16(weight_path):
  if weight_path is None:
    return models.vgg16(pretrained=True)
  else:
    vgg16 = models.vgg16(pretrained=False)
    vgg16_dict = vgg16.state_dict()
    weight = torch.load(weight_path)
    weight = {**vgg16_dict, **weight}
    vgg16.load_state_dict(weight)
    return vgg16

######################################
# IMAGE NETWORKS

class VGG16_RELU(nn.Module):

  def __init__(self, layer, weight_path=None):
    super(VGG16_RELU, self).__init__()
    vgg16 = load_vgg16(weight_path)
    self.features = nn.Sequential(*list(vgg16.features.children())[:layer])

  def forward(self, x):
    return self.features(x)


class VGG16_FC(nn.Module):

  def __init__(self, layer, weight_path=None):
    super(VGG16_FC, self).__init__()
    vgg16 = load_vgg16(weight_path)
    self.features = vgg16.features
    self.fc = nn.Sequential(*list(vgg16.classifier.children())[:layer])
    
  def forward(self, x):
    feat = self.features(x)
    feat = self.fc(feat.view(feat.size(0),-1))
    return feat

######################################
# ROI NETWORKS

class VGG16_RELU_ROI(nn.Module):

  def __init__(self, layer, wsize, rt, weight_path=None):
    super(VGG16_RELU_ROI, self).__init__()
    vgg16 = load_vgg16(weight_path)
    self.features = nn.Sequential(*list(vgg16.features.children())[:layer])
    self.RoI = ops.RoIPool((wsize,wsize),rt)

  def forward(self, x, rois):
    feat = self.features(x)
    feat = self.RoI(feat, rois)
    return feat


class VGG16_FC_ROI(nn.Module):

  def __init__(self, layer, wsize, rt, weight_path=None):
    super(VGG16_FC_ROI, self).__init__()
    vgg16 = load_vgg16(weight_path)
    self.features = nn.Sequential(*list(vgg16.features.children())[:30])
    self.fc = nn.Sequential(*list(vgg16.classifier.children())[:layer])
    self.RoI = ops.RoIPool((wsize,wsize),rt)

  def forward(self, x, rois): 
    feat = self.features(x)
    feat = self.RoI(feat, rois)
    feat = self.fc(feat.view(feat.size(0),-1))
    return feat


################################################################################
# FUNCTIONS

######################################
# GET IMAGE NETWORKS

def get_vgg16_relu(layer_name, weight_path=None):
  if layer_name == 'relu53':
    return VGG16_RELU(30, weight_path)
  elif layer_name == 'relu52':
    return VGG16_RELU(28, weight_path)
  elif layer_name == 'relu43':
    return VGG16_RELU(23, weight_path)
  elif layer_name == 'relu42':
    return VGG16_RELU(21, weight_path)
  elif layer_name == 'relu33':
    return VGG16_RELU(16, weight_path)
  else:
    raise Exception('layer_name %s not supported!'%layer_name)


def get_vgg16_fc(layer_name, weight_path=None):
  if layer_name == 'fc6':
    return VGG16_FC(2, weight_path)
  elif layer_name == 'fc7':
    return VGG16_FC(5, weight_path)
  else:
    raise Exception('layer_name %s not supported!'%layer_name)

def get_vgg16(layer_name, weight_path=None):
  if 'relu' in layer_name:
    return get_vgg16_relu(layer_name, weight_path)
  elif 'fc' in layer_name:
    return get_vgg16_fc(layer_name, weight_path)
  else:
    raise Exception('layer_name %s not supported!'%layer_name)


######################################
# GET ROI NETWORKS

def get_vgg16_relu_roi(layer_name, wsize, rt, weight_path=None):
  if layer_name == 'relu53':
    return VGG16_RELU_ROI(30, wsize, rt, weight_path)
  elif layer_name == 'relu43':
    return VGG16_RELU_ROI(23, wsize, rt, weight_path)
  elif layer_name == 'relu33':
    return VGG16_RELU_ROI(16, wsize, rt, weight_path)
  else:
    raise Exception('layer_name %s not supported!'%layer_name)


def get_vgg16_fc_roi(layer_name, wsize, rt, weight_path=None):
  if layer_name == 'fc6':
    return VGG16_FC_ROI(2, wsize, rt, weight_path)
  elif layer_name == 'fc7':
    return VGG16_FC_ROI(5, wsize, rt, weight_path)
  else:
    raise Exception('layer_name %s not supported!'%layer_name)

def get_vgg16_roi(layer_name, wsize, rt, weight_path=None):
  if 'relu' in layer_name:
    return get_vgg16_relu_roi(layer_name, wsize, rt, weight_path)
  elif 'fc' in layer_name:
    return get_vgg16_fc_roi(layer_name, wsize, rt, weight_path)
  else:
    raise Exception('layer_name %s not supported!'%layer_name)


######################################
# TRANSFORM IMAGES

def vgg_transform(img):
  """
  Parameters:

    img: (H x W x C) or (H x W) np array with values in range (0..255)
         or PIL Image instance.

  Returns:

    (3 x H x W) Tensor.

  """
  img = np.array(img,dtype=np.float32)
  if img.ndim == 2:
    img = np.repeat(img[:,:,np.newaxis], 3, axis=2)
  if img.ndim == 3 and img.shape[2] == 4:
    img = img[:,:,:3]
  img /= 255.0
  img -= np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
  img /= np.array([0.229, 0.224, 0.225]).reshape(1,1,3)
  img = torch.from_numpy(img.transpose((2,0,1)).copy().astype(np.float32))
  return img


