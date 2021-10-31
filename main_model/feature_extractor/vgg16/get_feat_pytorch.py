import torch
from torch import nn, Tensor

import pytorch_nets

import numpy as np
from scipy.io import loadmat, savemat
import h5py
from PIL import Image
import os
from os.path import join
import pickle
from tqdm import tqdm


##############################################################################

def read_imdb(imdb_path, large_file):
  if large_file:
    imdb = h5py.File(imdb_path, 'r')
  else:
    imdb = loadmat(imdb_path)
  return imdb

def read_img_imdb(args, clname):
  imdb_path = join(args.home, args.imgset, clname, 
                   clname + args.imgset_suffix + '.mat')
  imdb = read_imdb(imdb_path, args.imgset_h5py)
  if 'images' in imdb:
    num_images = np.prod(imdb['images'].shape)
  elif 'image_paths' in imdb: 
    num_images = np.prod(imdb['image_paths'].shape)
  else:
    raise Exception("imdb must have 'images' or 'image_paths' field!")

  return imdb, num_images

def read_img_and_prop_imdb(args, clname):
  img_imdb_path = join(args.home, args.imgset, clname, 
                       clname + args.imgset_suffix + '.mat')
  prop_imdb_path = join(args.home, args.propset, clname, 
                        clname + args.propset_suffix + '.mat')
  img_imdb, num_images = read_img_imdb(args, clname)
  if img_imdb_path == prop_imdb_path:
    prop_imdb = img_imdb
  else:
    prop_imdb = read_imdb(prop_imdb_path, args.propset_h5py)
    num_props = np.prod(prop_imdb['proposals'].shape)
    assert(num_props == num_images)
  
  return prop_imdb, img_imdb, num_images

def get_image(imdb, i, large_file=False, imagenet_resize=False):
  if 'images' in imdb:
    if large_file:
      img = np.array(imdb[imdb['images'][0][i]]).transpose(2,1,0)
    else:
      img = imdb['images'][i][0]
  elif 'image_paths' in imdb:
    if large_file:
      pass
    else:
      img = np.array(Image.open(imdb['image_paths'][i][0][0]))
  else:
    raise Exception("imdb must have 'images' or 'image_paths' field!")


  if img.ndim == 2:
    img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
  
  if img.ndim == 3 and img.shape[2] == 4:
    img = img[:,:,:3]
  
  img = Image.fromarray(img)
  if imagenet_resize:
    img = img.resize((224,224))
  return img

def get_rois(imdb, i, rois_type, large_file):
  """
  rois_type: string, 'bboxes' or 'proposals'
  """
  if large_file:
    rois = np.array(imdb[imdb[rois_type][0][i]]).T
  else:
    rois = np.array(imdb[rois_type][i][0], dtype=np.float32)
  rois[:, [0,1]] -= 1
  return rois

def get_net_image_features(args):
  net = pytorch_nets.get_vgg16(args.layer_name, args.weight_path).eval()
  if args.cuda: net = net.cuda()  
  return net

def get_net_roi_features(args):
  net = pytorch_nets.get_vgg16_roi(args.layer_name, args.wsize, 
                                   args.rt, args.weight_path).eval()
  if args.cuda: net = net.cuda() 
  return net

##############################################################################

def extract_image_features(args):
  classes = ['mixed']
  net = get_net_image_features(args)
  with torch.no_grad():
    for cl in args.class_indices:
      clname = classes[cl]
      imdb, n = read_img_imdb(args, clname)
      save_path = join(args.home, args.imgset, clname, 
                       args.save_loc, args.deepfeat_name)
      print('Processing for class %s with %d images'%(clname, n), flush=True)
      if args.save_result: 
        os.makedirs(save_path, exist_ok=True)
      print('Processing image', end=' ', flush=True)
      row_indices = np.unique(np.minimum(args.row_indices, n-1))
      for i in tqdm(row_indices):
        image = get_image(imdb,i,args.imgset_h5py,args.imagenet_resize)
        image =  pytorch_nets.vgg_transform(image).unsqueeze_(0)
        if args.cuda: image = image.cuda()
        data = {'data':net(image).detach().cpu().numpy()[0]}
        if args.save_result:
          savemat(join(save_path, '%d.mat'%(i+1)), data, do_compression=True)
        del data, image
        torch.cuda.empty_cache()
      print()


def extract_roi_features(args):
  classes = ['mixed']
  net = get_net_roi_features(args)
  with torch.no_grad():
    for cl in args.class_indices:
      clname = classes[cl]
      prop_imdb, img_imdb, n = read_img_and_prop_imdb(args, clname)
      save_path = join(args.home, args.propset, clname, args.save_loc, 
                       args.deepfeat_name)
      print('Processing for class %s with %d images'%(clname, n), flush=True)
      if args.save_result: 
        os.makedirs(save_path, exist_ok=True)
      print('Processing image', end=' ', flush=True)
      row_indices = np.unique(np.minimum(args.row_indices, n-1))
      for i in tqdm(row_indices):
        image = get_image(img_imdb, i, args.imgset_h5py)
        image =  pytorch_nets.vgg_transform(image).unsqueeze_(0)
        if args.cuda: image = image.cuda()
        props = get_rois(prop_imdb, i, args.rois_type, args.propset_h5py)
        rois = np.zeros([props.shape[0], 5], dtype=np.float32)
        rois[:,1:5] = props
        rois = Tensor(rois)
        if args.cuda: rois = rois.cuda()
        data = {'data':net(image, rois).detach().cpu().numpy()}
        if args.save_result:
          savemat(join(save_path, '%d.mat'%(i+1)), data, do_compression=True)
        del data, image
        torch.cuda.empty_cache()
      print()