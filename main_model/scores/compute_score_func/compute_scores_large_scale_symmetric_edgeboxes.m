function [] = compute_scores_large_scale_symmetric_edgeboxes(args, class_indices, row_indices)
% [] = compute_scores_large_scale_symmetric_edgeboxes(args, class_indices, row_indices)
%
% Function to compute confidence/standout scores.
%
% Parameters:
% 
%   args: struct, field are cell arrays, which are
%
%     root: string, root folder of the image set.
%
%     clname: string, name of the class.
%
%     small_imdb: bool, 
%
%     saveconf: bool, whether to save confidence scores.
%
%     savestd: bool, whether to save standout scores.
%
%   ---------------------
%
%     PHM_type: string, type of PHM function.
%
%     symmetric: bool, whether to symmetrize score matrices.
%
%     num_pos: int, number of positive entries in score matrices
%                         after sparsification.
%
%     num_pos_text: string, text code for num_pos.
%
%     stdver: int, version of the standout function.
%
%     max_iter: int, max_iteration in computing standout.
%
%     area_ratio: double, area ratio in the definition of background and 
%                    context regions.
%
%     area_ratio_text: string, text code for area_ratio.
%
%   ---------------------
%
%     deepfeat: bool, whether to use CNN features in the score computation.
%
%     feat_type: string, name of the features.
%
%     sim_type: string, text code for similarity function.
%
%   ---------------------
%
%     prefiltered_nb: bool, whether to only compute scores between an image
%                        and its neighbor candidates.
%
%     num_nb: int, number of neighbors of each image.
%
%     nb_root: string, root folder containing image pre-neighbors.
%
%     nb_type: string, type of the neighbors. 
%
% Returns: 
%
%

args = argument_reader(args);
for cl = class_indices
  opts = get_opts(args, cl);
  [msg, err] = argument_checker(opts);
  fprintf('%s\n', msg);
  if err 
    return;
  end
  opts.cl = cl;
  fprintf('Processing for class %s\n', opts.clname);
  %-------------------------------
  % load imdb
  [opts.images_size, opts.prop_path] = load_imdb(opts);
  opts.n = size(opts.images_size, 1);
  opts.PL = ProposalLoaderEB(opts.prop_path, opts.n, opts.prop_step);
  opts.indices = build_indices(opts);
  opts.feat_path = fullfile(opts.feat_root, opts.clname, 'features/proposals', opts.feat_type);
  [opts.save_path_conf, opts.save_path_std] = get_save_paths(opts);          
  %-------------------------------
  % compute scores
  S_confidence = {};
  S_standout = {};

  cache_i = 0; cache_j = 0;
  feati = []; featj = [];
  proposals_i = []; proposals_j = [];

  for row = row_indices
    time_zero = tic;
    i = opts.indices(row, 1); j = opts.indices(row, 2);
    if ~isequal(cache_i, i)
      [proposals_i, scores_i] = ProposalLoaderEB_load(opts.PL, i);
      feati = read_feat(opts, i) .* scores_i(:);
      cache_i = i;
    end
    if ~isequal(cache_j, j)
      [proposals_j, scores_j] = ProposalLoaderEB_load(opts.PL, j);
      featj = read_feat(opts, j) .* scores_j(:);
      cache_j = j;
    end
    [confidence, standout] = compute_scores_(opts, feati, featj, proposals_i, proposals_j, i, j);
    info = whos;
    fprintf('Memory: %.2f Gb\n', sum(cell2mat({info.bytes}))/1024^3);

    if opts.saveconf
      S_confidence{row} = confidence;
    end
    if opts.savestd
      S_standout{row} = standout;
    end
    fprintf('Scores computed in %.4f secs..........................\n', toc(time_zero));
  end
  % save result
  fprintf('Results will be saved to: \n%s\n%s\n', opts.save_path_conf, opts.save_path_std);
  if opts.saveconf
    S_confidence = S_confidence(row_indices(1):row_indices(end));
  end
  if opts.savestd
    S_standout = S_standout(row_indices(1):row_indices(end));
  end
  save_results(opts, min(row_indices), max(row_indices), S_confidence, S_standout);
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LOCAL FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [args] = argument_reader(args)
  assert(all(isfield(args, {...
                        'root', 'clname', 'small_imdb', 'saveconf', 'savestd', ...
                        'PHM_type', 'symmetric', 'num_pos', 'num_pos_text', ...
                        'stdver', 'max_iter', 'area_ratio', 'area_ratio_text', ...
                        'deepfeat', 'feat_type', 'sim_type', ...
                        'prefiltered_nb', 'nb_root', 'nb_type', 'num_nb', ...
                            })));
  num_classes = numel(args.root);
  for i = 1:num_classes
    if strcmp(args.PHM_type{i}, '')
      args.PHM_func{i} = @PHM_lite;
    elseif strcmp(args.PHM_type{i}, 'max')
      args.PHM_func{i} = @PHM_lite_max;
    elseif strcmp(args.PHM_type{i}, 'sum')
      args.PHM_func{i} = @PHM_lite_sum;
    elseif strcmp(args.PHM_type{i}, 'tmp')
      args.PHM_func{i} = @PHM_lite_tmp;
    else 
      error('Unknown PHM function!');
    end

    if args.savestd{i}
      if args.stdver{i} == 4
        args.stdfunc{i} = @standout_box_pair_v4;
      else 
        error('Version of standout function not supported!');
      end
    end

    args.scname{i} = args.feat_type{i};
    if args.deepfeat{i}
      args.scname{i} = [args.scname{i}, '_', args.sim_type{i}];
    end
    if args.symmetric{i}
      args.scname{i} = [args.scname{i}, '_symmetric'];
    end 
    if ~strcmp(args.PHM_type{i}, '') 
      args.scname{i} = [args.scname{i}, '_', args.PHM_type{i}];
    end
    if args.prefiltered_nb{i}
      for neighbor_idx = 1:numel(args.nb_type{i})
        args.scname{i} = sprintf('%s_%d_%s', args.scname{i}, ...
                                             args.num_nb{i}, ...
                                             args.nb_type{i}{neighbor_idx});
      end
    end
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [opts] = get_opts(args, cl)
  opts = struct;
  fields = fieldnames(args);
  for i = 1:numel(fields)
    fieldvalues = getfield(args, fields{i});
    opts = setfield(opts, fields{i}, fieldvalues{cl});
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [msg, err] = argument_checker(args)
  msg = 'Message from checker: ';
  err = false;
  if (args.small_imdb & (numel(args.feat_type) < 6 | ~strcmp(args.feat_type(1:6), 'small_'))) | ...
     (~args.small_imdb & (numel(args.feat_type) >= 6 & strcmp(args.feat_type(1:6), 'small_')))
      msg = sprintf('%s\n\t%s', msg, 'feat_type and small_imdb are not compatible');
      err = true;
  end

  if (args.num_pos == Inf & ~strcmp(args.num_pos_text, 'full')) | ...
     (args.num_pos < Inf & args.num_pos ~= str2double(args.num_pos_text))
    msg = sprintf('%s\n\t%s', msg, 'num_pos and num_pos_text are not compatible');
    err = true;
  end

  if args.deepfeat & ~any(cellfun(@(x) strcmp(x, args.sim_type) == 1, ...
                                  {'sp', 'cos', 'spatial_cos', '01', 'spatial_01', 'sqrt', 'log'}))
    msg = sprintf('%s\n\t%s', msg, 'sim_type must be in {"sp", "cos", "01", "sqrt", "log"}');
    err = true;
  end

  msg = sprintf('%s\nEnd message.\n', msg);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [images_size, prop_path] = load_imdb(opts)
  if opts.small_imdb 
    imdb = load(fullfile(opts.root, opts.clname, [opts.clname, '_small_lite.mat']), 'images_size');
  else 
    imdb = load(fullfile(opts.root, opts.clname, [opts.clname, '_lite.mat']), 'images_size');
  end
  images_size = imdb.images_size;
  prop_path = fullfile(opts.root, opts.clname, [opts.clname, '_proposals']);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [e] = load_neighbors_single_(opts, idx)
  nb_path = fullfile(opts.nb_root{idx}, opts.nb_type{idx}, opts.clname, ...
                     sprintf('%d.mat', opts.num_nb));
  e = getfield(load(nb_path), 'e');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [e] = load_neighbors_multiple(opts)
  ids = 1:numel(opts.nb_root);
  e = load_neighbors_single_(opts, ids(1));
  for idx = ids(2:end)
    current_e = load_neighbors_single_(opts, idx);
    e = arrayfun(@(i) [e{i}, current_e{i}], [1:size(current_e,1)]', 'Uni', false);
  end 
  e = cellfun(@(x) unique(x), e, 'Uni', false);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [indices] = build_indices(opts)
  n = opts.n; 
  if opts.prefiltered_nb
    e = load_neighbors_multiple(opts);
    assert(size(e,1) == n, sprintf('Size of e is %d\n', size(e)));
    actual_num_nb = cellfun(@numel, e);
    indices = [repelem([1:n]', actual_num_nb) reshape(cell2mat(e'), [], 1)];
    indices = [indices; indices(:,[2,1])];
    indices = unique(indices, 'rows');
  else 
    indices = [repelem([1:n]',n,1) repmat([1:n]',n,1)];
    indices = indices(indices(:,1) ~= indices(:,2), :);
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [save_path_conf, save_path_std] = get_save_paths(opts)
  save_path_conf = fullfile(opts.root, opts.clname, 'confidence_symmetric_edgeboxes', ...
                            sprintf('%s_normalized_%s', opts.scname, opts.num_pos_text));
  save_path_std = fullfile(opts.root, opts.clname, 'standout_symmetric_edgeboxes', ...
                           sprintf('%s_v%d_%s_normalized_%s', ...
                                   opts.scname, opts.stdver, ...
                                   opts.area_ratio_text, opts.num_pos_text));  
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [feati] = read_feat(opts, i)
  h = tic;
  feati = getfield(load(fullfile(opts.feat_path, sprintf('%d.mat', i))), 'data');
  if opts.deepfeat
    feati = process_deepfeat(feati, opts.sim_type);
  end
  assert(numel(size(feati)) == 2);
  fprintf('Load feat: %f\n',toc(h));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [confidence, standout] = compute_scores_(opts, feati, featj, proposals_i, proposals_j, i, j)
  timer_conf = tic;
  if opts.symmetric
    score1 = PHM_confidence_lite(opts.PHM_func, ...
                                 opts.images_size{i}, opts.images_size{j}, ...
                                 proposals_i, proposals_j, ...
                                 feati, featj, 'RAW');
    score2 = PHM_confidence_lite(opts.PHM_func, ...
                                 opts.images_size{j}, opts.images_size{i}, ...
                                 proposals_j, proposals_i, ...
                                 featj, feati, 'RAW');
    confidence = max(score1, 0) + max(score2', 0);
  else 
    score = PHM_confidence_lite(opts.PHM_func, ...
                                opts.images_size{i}, opts.images_size{j}, ...
                                proposals_i, proposals_j, ...
                                feati, featj, 'RAW');
    confidence = max(score, 0);
  end
  confidence = confidence/prod(size(confidence))*1e6;
  fprintf('Confidence computed in %f sec\n', toc(timer_conf));
  % compute and sparsify standout
  timer_std = tic;
  if opts.savestd
    standout = opts.stdfunc(transpose(proposals_i), transpose(proposals_j), ...
                            confidence, opts.max_iter, opts.area_ratio);
    if opts.num_pos ~= Inf
      standout = sparsify_matrix(standout, opts.num_pos);
    else 
      standout = sparse(double(standout));
    end
  else 
    standout = [];
  end
  fprintf('Standout computed in %f secs\n', toc(timer_std));
  % sparsify confidence
  if opts.saveconf
    if opts.num_pos ~= Inf
      confidence = sparsify_matrix(confidence, opts.num_pos);
    else 
      confidence = sparse(double(confidence));
    end
  else 
    confidence = [];
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [] = save_results(opts, left, right, ...
                           S_confidence, S_standout)
  if opts.saveconf
    mkdir(opts.save_path_conf);
    if exist(fullfile(opts.save_path_conf, 'indices.mat'), 'file') ~= 2
      indices = opts.indices;
      save(fullfile(opts.save_path_conf, 'indices.mat'), 'indices');
    else 
      fprintf('Indices.mat in confidence save path has been created!\n');
    end
    fprintf('Saving confidence score ...\n');
    data.data = S_confidence;
    savefile(fullfile(opts.save_path_conf, sprintf('%d_%d.mat', left, right)), data);
  end

  if opts.savestd
    mkdir(opts.save_path_std);
    if exist(fullfile(opts.save_path_std, 'indices.mat'), 'file') ~= 2
      indices = opts.indices;
      save(fullfile(opts.save_path_std, 'indices.mat'), 'indices');
    else 
      fprintf('Indices.mat in standout save path has been created!\n');
    end
    fprintf('Saving standout score ...\n');
    data.data = S_standout;
    savefile(fullfile(opts.save_path_std, sprintf('%d_%d.mat', left, right)), data);
  end
end
