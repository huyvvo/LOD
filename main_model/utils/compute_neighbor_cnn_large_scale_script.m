DATA_ROOT = '~/data';
similarity_measure = 'cos';
num_neighbors = 100;
num_parts = 12;

imgset = 'opiv6_200k/opiv6_200k';
feat_name = 'vgg16_fc6_resize';
class_indices = 1;

root = fullfile(DATA_ROOT, imgset);
classes = get_classes(imgset);

for cl = class_indices
  clname = classes{cl};
  fprintf('Processing for class %s\n', clname);
  imdb = load(fullfile(root, clname, [clname, '_lite.mat']));
  n = numel(imdb.bboxes);

  feats = cell(n,1);
  fprintf('Loading features ...\n');
  load_tic = tic;
  parfor i = 1:n 
    feat = getfield(load(fullfile(root, clname, 'features/image', feat_name, sprintf('%d.mat', i))), 'data');
    if size(feat,1) == 1
      feat = feat';
    end
    feats{i} = mean(reshape(feat, size(feat,1), []), 2)';
  end
  fprintf('\n');
  fprintf('Features loaded in %.2f\n', toc(load_tic));
  feats = single(cell2mat(feats));

  e = cell(n,1);
  parts = arrayfun(@(pidx) round(n/num_parts*(pidx-1)+1):round(n/num_parts*pidx), 1:num_parts, 'Uni', false);
  neighbor_tic = tic;
  if strcmp(similarity_measure, 'cos')
    norm_feats = feats ./ sqrt(sum(feats.*feats, 2));
    fprintf('Computing neighbors ')
    for pidx = 1:num_parts 
      first_index = parts{pidx}(1);
      second_index = parts{pidx}(2);
      current_sim = norm_feats(first_index:second_index,:) * norm_feats';
      current_sim(i) = -1;
      [~, max_idx] = sort(current_sim, 'descend');
      e{i} = max_idx(1:num_neighbors);
      if ismember(i, e{i})
        e{i} = setdiff(e{i}, i);
      end
    end
    fprintf('\n');
  end
  fprintf('Neighbors found in %.2f\n', toc(neighbor_tic));
  
  save_dir = fullfile(root, sprintf('neighbor_%s_%s/%s', similarity_measure, feat_name, clname));
  mkdir(save_dir);
  save_path = fullfile(save_dir, sprintf('%d.mat', num_neighbors));
  fprintf('Saving results to %s\n', save_path);
  save(save_path, 'e');

end