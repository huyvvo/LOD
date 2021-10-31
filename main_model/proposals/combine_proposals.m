function [stats] = combine_proposals(imgset, proposal_names, class_indices, varargin)
%
% [] = combine_proposals(imgset, proposal_names, class_indices, varargin)
% 
% Default arguments:
% DATA_ROOT = '~/data';
% save_result = true;
% save_suffix = '';
% verbose = true;

% Default arguments
DATA_ROOT = '~/data';
save_result = true;
save_suffix = '';
verbose = true;

varnames = varargin(1:2:length(varargin));
varvals = varargin(2:2:length(varargin));
validnames = {'DATA_ROOT', 'save_suffix', 'verbose', 'save_result'};
for name = varnames
  if ~any(cellfun(@(el) strcmp(name{:}, el), validnames))
    error(sprintf('"%s" is not a valid argument!', name{:}));
  end
end
for name = validnames
  if ~isempty(strmatch(name{:},varnames))
    evalc(sprintf('%s=varvals{strmatch(name{:},varnames)}',name{:}));
  end
end

fprintf('Arguments:\n');
fprintf('DATA_ROOT: %s\n', DATA_ROOT);
fprintf('imgset: %s\n', imgset);
fprintf('proposal_names: '); cellfun(@(el) fprintf('%s, ', el), proposal_names, 'Uni', false); fprintf('\n');
fprintf('class indices: %d ', class_indices); fprintf('\n');
fprintf('save_result: %s\n', string(save_result));
fprintf('save_suffix: %s\n', save_suffix);
fprintf('verbose: %s\n', string(verbose));

%-----------------------------------------------------

stats = cell(numel(class_indices),1);

root = fullfile(DATA_ROOT, imgset);
classes = get_classes(imgset);
fprintf('DATA_ROOT: %s\n', root);
for cl = class_indices
  clname = classes{cl};
  fprintf('Processing for class %s\n', clname);
  if save_result
    save_path = fullfile(root, clname, [clname, save_suffix, '_all_lite.mat']);
  end

  imdb_paths = cellfun(@(f) fullfile(root, clname, [clname, '_', f, '.mat']), proposal_names, 'Uni', false);
  imdbs = {};
  for imdb_idx = 1:numel(imdb_paths)
    imdbs{imdb_idx} = load(imdb_paths{imdb_idx});
  end
  assert(all(cellfun(@(x) isequal(imdbs{1}.bboxes, x.bboxes), imdbs)));
  n = size(imdbs{1}.bboxes, 1);

  fields = fieldnames(imdbs{1});
  imdb = struct;
  for fidx = 1:numel(fields)
    imdb = setfield(imdb, fields{fidx}, getfield(imdbs{1}, fields{fidx}));
  end
  for i = 1:n 
    for imdb_idx = 2:numel(imdb_paths)
      imdb.proposals{i} = [imdb.proposals{i} ; imdbs{imdb_idx}.proposals{i}];
      imdb.root_ids{i} = [imdb.root_ids{i} ; imdbs{imdb_idx}.root_ids{i}];
      if isfield(imdb, 'root_feat_types')
        imdb.root_feat_types{i} = [imdb.root_feat_types{i} ; imdbs{imdb_idx}.root_feat_types{i}];
      end
      if isfield(imdb, 'root_feat_types_code')
        imdb.root_feat_types_code{i} = [imdb.root_feat_types_code{i} ; imdbs{imdb_idx}.root_feat_types_code{i}];
      end
    end
    [imdb.proposals{i}, unique_ids] = unique(imdb.proposals{i}, 'rows');
    imdb.root_ids{i} = imdb.root_ids{i}(unique_ids);
    if isfield(imdb, 'root_feat_types')
      imdb.root_feat_types{i} = imdb.root_feat_types{i}(unique_ids);
    end
    if isfield(imdb, 'root_feat_types_code')
      imdb.root_feat_types_code{i} = imdb.root_feat_types_code{i}(unique_ids);
    end
  end

  if save_result
    savefile(save_path, imdb);
  end

  num_props = [];
  corloc05 = [];
  corloc07 = [];
  corloc09 = [];
  numpos05 = [];
  numpos07 = [];
  numpos09 = [];
  pos_box_percentage = [];
  mean_num_box_retrieved = [];
  sum_num_box_retrieved = [];
  percentage_box_retrieved = [];
  num_boxes = [];

  proposals = imdb.proposals;
  bboxes = imdb.bboxes;

  num_pos_boxes = count_valid_boxes(proposals, bboxes, create_x_all_regions(proposals), 0.5)';
  num_proposals = cellfun(@numel, proposals)/4;

  num_objects = cellfun(@(x) size(x,1), bboxes);
  [~, iou] = CorLoc(proposals, bboxes, create_x_all_regions(proposals), 0.5);
  num_objects_retrieved = cellfun(@(x) sum(max(x') >= 0.5), iou);

  num_props = [num_props mean(num_proposals)];
  corloc05 = [corloc05 CorLoc(proposals, bboxes, create_x_all_regions(proposals), 0.5)];
  corloc07 = [corloc07 CorLoc(proposals, bboxes, create_x_all_regions(proposals), 0.7)];
  corloc09 = [corloc09 CorLoc(proposals, bboxes, create_x_all_regions(proposals), 0.9)];
  numpos05 = [numpos05 mean(count_valid_boxes(proposals, bboxes, create_x_all_regions(proposals), 0.5))];
  numpos07 = [numpos07 mean(count_valid_boxes(proposals, bboxes, create_x_all_regions(proposals), 0.7))];
  numpos09 = [numpos09 mean(count_valid_boxes(proposals, bboxes, create_x_all_regions(proposals), 0.9))];
  pos_box_percentage = [pos_box_percentage mean(num_pos_boxes./num_proposals')*100];
  mean_num_box_retrieved = [mean_num_box_retrieved mean(num_objects_retrieved)];
  sum_num_box_retrieved = [sum_num_box_retrieved sum(num_objects_retrieved)];
  percentage_box_retrieved = [percentage_box_retrieved mean(num_objects_retrieved./num_objects)];
  num_boxes = [num_boxes sum(num_objects)];

  stats{cl} = [pos_box_percentage(end), sum(num_objects_retrieved) num_props(end)];

  if verbose
    fprintf('Average number of proposals: %.2f\n', num_props(end));
    fprintf('CorLoc 05/07/09: %.4f/%.4f/%.4f\n', corloc05(end), corloc07(end), corloc09(end));
    fprintf('Number of positive boxes 05/07/09: %.2f/%.2f/%.2f\n', numpos05(end), numpos07(end), numpos09(end));
    fprintf('Percentage of positive boxes: %.2f%%\n', pos_box_percentage(end));
    fprintf('Average number of objects retrieved: %.2f\n', mean_num_box_retrieved(end));
    fprintf('Number of objects retrieved: %d\n', sum_num_box_retrieved(end));
    fprintf('Average percentage of objects retrieved: %.2f\n', percentage_box_retrieved(end));
    fprintf('Total number of objects: %d\n', num_boxes(end));
  end

end
