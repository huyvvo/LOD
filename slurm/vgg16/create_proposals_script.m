function script(START, END, NUM_IMAGES, DATASET_NAME, LOD_ROOT)
  script_aux(START, END, NUM_IMAGES, DATASET_NAME, LOD_ROOT, 'vgg16_relu43');
  script_aux(START, END, NUM_IMAGES, DATASET_NAME, LOD_ROOT, 'vgg16_relu53');
end

function [] = script_aux(START, END, NUM_IMAGES, DATASET_NAME, LOD_ROOT, FEAT_TYPE)

if START > NUM_IMAGES
  fprintf('START > NUM_IMAGES, do nothing!\n');
  return
end
fprintf('Arguments:\n')
fprintf('START: %s\n', string(START));
fprintf('END: %s\n', string(END));
fprintf('NUM_IMAGES: %s\n', string(NUM_IMAGES));

END = min(END, NUM_IMAGES);
fprintf('END adjusted to %s\n', string(END));

cd(LOD_ROOT)
setup;

ALPHA = 0.3; BETA_GLOBAL = 0.5; ROOT_THRESHOLD = 1.0;
class_indices = 1;
row_indices = START:END;

DATA_ROOT = fullfile(LOD_ROOT, 'data');
SAVE_ROOT = fullfile(LOD_ROOT, 'data');

all_classes = 1:1;
args = struct;

%--------------------------------------------------------------------------------------

alpha = ALPHA;
num_maxima = 20;
ws = 3;
conn = 4;
beta_global = BETA_GLOBAL;
beta_local = 1;
num_levels = 50;
root_threshold = ROOT_THRESHOLD;

[args.alpha{all_classes}] = deal(alpha);
[args.num_maxima{all_classes}] = deal(num_maxima);
[args.ws{all_classes}] = deal(ws);
[args.conn{all_classes}] = deal(conn);
[args.beta_global{all_classes}] = deal(beta_global);
[args.beta_local{all_classes}] = deal(beta_local);
[args.num_levels{all_classes}] = deal(num_levels);
[args.root_threshold{all_classes}] = deal(root_threshold);

%--------------------------------------------------------------------------------------

imgset = DATASET_NAME;
small_imdb = false;
feat_type = FEAT_TYPE;
save_dir = sprintf('%s_vgg16_cnn_%02d_%d_%02d_%02d_%02d_with_roots', ...
                   imgset, 10*alpha, num_maxima,10*beta_local, 10*beta_global, 10*root_threshold);
save_result = true;

[args.DATA_ROOT{all_classes}] = deal(DATA_ROOT);
[args.root{all_classes}] = deal(fullfile(DATA_ROOT, imgset));
[args.small_imdb{all_classes}] = deal(small_imdb);
[args.feat_type{all_classes}] = deal(feat_type);
[args.save_dir{all_classes}] = deal(save_dir);
[args.save_result{all_classes}] = deal(save_result);

args.clname = get_classes(imgset);
args.save_root = fullfile(SAVE_ROOT, args.save_dir);

%----------------------------------------------------------------------------------------

[proposals, root_ids, root_feat_types] = create_proposals_cnn(args, class_indices, row_indices);

end