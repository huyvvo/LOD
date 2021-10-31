from argparse import Namespace, ArgumentParser
import tools


parser = ArgumentParser()
parser.add_argument('--layer_name', '-l', type=str, required=True,  
                    help=f'Layer to extract features. Valid values are "relu33", '
                         f'"relu42", "relu43", "relu52", "relu53" and "fc6".')
parser.add_argument('--resize_image', '-rs', type=lambda x: x.lower()=='true', required=True,
                    help='Whether to resize image to (224,224)')
parser.add_argument('--cuda', type=lambda x: x.lower()=='true', required=True, 
                    help='Whether to use GPUs')
parser.add_argument('--class_indices', '-cl', type=int, required=True, action='append', 
                    help='Class indices')
parser.add_argument('--START', type=int, required=True, help='First image index')
parser.add_argument('--END', type=int, required=True, help='Last image index')

parser.add_argument('--home', type=str, required=True)
parser.add_argument('--imgset', type=str, required=True)
parser.add_argument('--imgset_suffix', type=str, required=True)

args = parser.parse_args()

opts = Namespace(
  layer_name = args.layer_name,
  imagenet_resize = args.resize_image, 
  cuda = args.cuda,

  home = args.home,
  imgset = args.imgset,
  imgset_suffix = args.imgset_suffix, # '' or '_image_paths'
)

tools.extract_image_feats(opts, args.class_indices, range(args.START, args.END))