function [] = proposal_stats(imgset, class_indices, varargin)
%
% [] = proposal_stats(imgset, class_indices, varargin)]
% 
% Default parameters:
%
%   DATA_ROOT = '~/data';
%   suffix = '_lite';
%   verbose = true;


% default arguments
DATA_ROOT = '~/data';
suffix = '_lite';
verbose = true;

varnames = varargin(1:2:length(varargin));
varvals = varargin(2:2:length(varargin));
validnames = {'DATA_ROOT', 'suffix', 'verbose'};
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
fprintf('imgset: %s\n', imgset);
fprintf('DATA_ROOT: %s\n', DATA_ROOT);
fprintf('suffix: %s\n', suffix);
fprintf('verbose: %s\n', string(verbose));

root = fullfile(DATA_ROOT, imgset);
classes = get_classes(imgset);
fprintf('---------------------------------------------------------\n');

num_props = [];
corloc05 = []; corloc07 = []; corloc09 = [];
numpos05 = []; numpos07 = []; numpos09 = [];
pos_box_percentage = [];
mean_num_box_retrieved = [];
sum_num_box_retrieved = [];
percentage_box_retrieved = [];
num_boxes = [];
for cl = class_indices
  clname = classes{cl};
  imdb = load(fullfile(root, clname, [clname, suffix, '.mat']));
  proposals = imdb.proposals;
  bboxes = imdb.bboxes;
  num_proposals = cellfun(@numel, proposals)/4;
  num_objects = cellfun(@numel, bboxes)/4;

  [corloc05(end+1),~,~,num_pos_regions05,num_objects_retrieved] = all_metrics(proposals, bboxes, create_x_all_regions(proposals), 0.5);
  [corloc07(end+1),~,~,num_pos_regions07,~] = all_metrics(proposals, bboxes, create_x_all_regions(proposals), 0.7);
  [corloc09(end+1),~,~,num_pos_regions09,~] = all_metrics(proposals, bboxes, create_x_all_regions(proposals), 0.9);

  numpos05(end+1) = mean(num_pos_regions05);
  numpos07(end+1) = mean(num_pos_regions07);
  numpos09(end+1) = mean(num_pos_regions09);

  num_props(end+1) = mean(num_proposals);
  pos_box_percentage(end+1) = mean(numpos05./num_proposals)*100;
  mean_num_box_retrieved(end+1) = mean(num_objects_retrieved);
  sum_num_box_retrieved(end+1) = sum(num_objects_retrieved);
  percentage_box_retrieved(end+1) = mean(num_objects_retrieved./num_objects);
  num_boxes(end+1) = sum(num_objects);
  
  
  if verbose
    fprintf('Processing for class %s\n', clname);
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
fprintf('---------------------------------------------------------\n');
fprintf('Average num_props: %.4f\n', mean(num_props));
fprintf('Average corloc05/07/09: %.4f/%.4f/%.4f\n', mean(corloc05), mean(corloc07), mean(corloc09));
fprintf('Average numpos05/07/09: %.4f/%.4f/%.4f\n', mean(numpos05), mean(numpos07), mean(numpos09));
fprintf('Average percentage pos boxes: %.2f%%\n', mean(pos_box_percentage));
fprintf('Total number of objects retrieved: %d\n', sum(sum_num_box_retrieved));
fprintf('Average percentage of objects retrieved: %.2f\n', mean(percentage_box_retrieved));
fprintf('Total number of objects: %d\n', sum(num_boxes));
