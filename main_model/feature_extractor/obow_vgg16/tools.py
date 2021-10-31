from argparse import Namespace
import get_feat_pytorch

def merge_configs(args, opts):
  """
  Replace the values of the arguments in 'args' with given 
  values in 'opts'. Arguments in 'opts' that are not in 'args' 
  are ignored.

  Parameters:

    args: Namespace, preset arguments.
    opts: Namespace, User-input arguments.

  Returns:
    
    A new Namespace

  """
  return Namespace(**{**vars(args), **vars(opts)})

def extract_image_feats(opts, class_indices, row_indices):
  args = Namespace()
  
  args.save_result = True
  args.cuda = True
  args.weight_path = 'torchvision_obow_vgg_model_net_checkpoint_80.pth.tar'

  args.layer_name = None # e.g. relu43, relu53 for vgg, layer2, layer3, ..., avgpool for resnet
  args.imagenet_resize = None # whether to resize image to [224,224] before feeding it to the network

  args.home = None
  args.imgset = None
  args.imgset_suffix = None # '' or '_small' for small dataset or '_image_paths' large dataset
  args.imgset_h5py = False

  args.save_loc = 'features/image' # 'features/proposals', 'features/image' or 'features/tmp'
  args.class_indices = class_indices
  args.row_indices = row_indices

  args.chmod = 0o664

  # merge configs
  args = merge_configs(args, opts)
  print(args)

  # process configs
  if 'checkpoint_80' in args.weight_path:
    args.deepfeat_name = f'vgg16_bn_obow80_{args.layer_name}'
  elif 'checkpoint_100' in args.weight_path:
    args.deepfeat_name = f'vgg16_bn_obow100_{args.layer_name}'
  else:
    raise Exception('weight_path does not specify checkpoint')

  if args.imagenet_resize:
    args.deepfeat_name += '_resize'

  args_dict = vars(args)
  for att in args_dict:
    print('%s:'%att, args_dict[att])

  get_feat_pytorch.extract_image_features(args)


def extract_roi_feats(opts, class_indices, row_indices):
  args = Namespace()
  args.save_result = True
  args.cuda = True
  args.weight_path = 'torchvision_obow_vgg_model_net_checkpoint_80.pth.tar'

  args.layer_name = 'relu53' # e.g. relu53 for vgg, layer3 for resnet
  args.deepfeat_name = 'vgg16_bn_obow80_relu53_77_roi_pooling_noresize'

  args.home = None
  args.propset = None
  args.propset_suffix = None # '', '_small' or '_lite'
  args.propset_h5py = None

  args.imgset = None
  args.imgset_suffix = None # '' or '_small' for small dataset or '_image_paths' large dataset
  args.imgset_h5py = False

  args.wsize = 7
  args.rt = 1/16

  args.save_loc = 'features/proposals' # 'features/proposals', 'features/image' or 'features/tmp'
  args.rois_type = 'proposals'
  args.class_indices = class_indices
  args.row_indices = row_indices

  args.chmod = 0o664

  args = merge_configs(args, opts)

  args_dict = vars(args)
  for att in args_dict:
    print('%s:'%att, args_dict[att])

  get_feat_pytorch.extract_roi_features(args)