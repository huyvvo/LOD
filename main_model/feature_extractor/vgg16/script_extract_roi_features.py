from argparse import Namespace, ArgumentParser
import tools


parser = ArgumentParser()
parser.add_argument('--cuda', type=lambda x: x.lower()=='true', required=True, 
                    help='Whether to use GPUs')
parser.add_argument('--class_indices', '-cl', type=int, required=True, action='append', 
                    help='Class indices')
parser.add_argument('--START', type=int, required=True, help='First image index')
parser.add_argument('--END', type=int, required=True, help='Last image index')

parser.add_argument('--home', type=str, default='/scratch/vv25/data/', help='Home data')
parser.add_argument('--propset', type=str, required=True)
parser.add_argument('--propset_suffix', type=str, default='_lite')
parser.add_argument('--propset_h5py', type=lambda x: x.lower()=='true', required=True)
parser.add_argument('--imgset', type=str, required=True)
parser.add_argument('--imgset_suffix', type=str, required=True)

parser.add_argument('--layer_name', type=str, required=True)
parser.add_argument('--deepfeat_name', type=str, required=True)

args = parser.parse_args()

opts = Namespace(
  cuda = args.cuda,

  home = args.home,
  propset = args.propset,
  propset_suffix = args.propset_suffix,
  propset_h5py = args.propset_h5py,

  layer_name = args.layer_name,
  deepfeat_name = args.deepfeat_name,

  imgset = args.imgset,
  imgset_suffix = args.imgset_suffix, # '' or '_image_paths'
)

tools.extract_roi_feats(opts, args.class_indices, range(args.START, args.END))