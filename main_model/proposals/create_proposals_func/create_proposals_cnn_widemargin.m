function [proposals, root_ids, root_feat_types] = create_proposals_from_cnn(args, class_indices, row_indices)
% CREATE_PROPOSALS_FROM_CNN
% [proposals, root_ids, root_feat_types] = create_proposals_from_cnn(args, class_indices, row_indices)
% 
% Create region proposals from local maxima in CNN saliency maps.
% 
% Parameters:
% 
%   args: struct, each field is a cell array, which are
% 
%     root: string, root folder of the image set.
% 
%     clname: string, name of the class.
% 
%     small_imdb: bool.
%  
%     feat_type: string, name of the CNN features used.
% 
%     -----------------------
% 
%     save_root: string, path to the folder to which the result will be saved.
%
%     save_result: bool, whether to save results.
% 
%     alpha: double, only local maxima whose saliency is greater than 
%            alpha * max saliency is retained.
% 
%     num_maxima: int, number of local maxima.
% 
%     ws: int, window size to identify the neighborhood of each local maximum.
% 
%     conn: int, connectivity in the graph of locations (4 or 8).
% 
%     beta_global: double, constant used for thresholding on the global 
%                  saliency map.
% 
%     beta_local: double, constant used for thresholding on the local 
%                 saliency map.
% 
%     num_levels: int, number of levels used to create proposals in the 
%                 local saliency map.
% 
%   -----------------------
% 
%   class_indices: array, indices of image classes to compute proposals.
%
%   row_indices: array, indices of images to compute proposals.
% 
% Returns:
% 
% 

args = argument_reader(args);
for cl = class_indices
  opts = get_opts(args, cl);
  fprintf('Processing for class %s\n', opts.clname);
  %---------------------------
  % load imdb
  [opts.bboxes, opts.imgsize] = load_imdb(opts);
  opts.n = numel(opts.bboxes);
  opts.spsize = cell(opts.n,1);
  %---------------------------
  % create proposals
  proposals = cell(opts.n,1);
  root_ids = cell(opts.n,1);
  root_feat_types = cell(opts.n,1);

  row_indices = row_indices(row_indices <= opts.n);
  tic;
  for i = row_indices
    if mod(i,10) == 1
      fprintf('Processed %d images in %.2f secs\n', i-1, toc);
    end
    [opts.feat, opts.spsize{i}] = load_feat(opts, i);
    raw_salc = sum(opts.feat');
    opts.feat = opts.feat./sqrt(sum(opts.feat.*opts.feat, 2));
    topo_salc = clean_salc_topo(raw_salc, opts.spsize{i}, opts.alpha);
    opts.peaks = topo_peaks(topo_salc, opts.num_maxima, opts.ws, opts.alpha*max(topo_salc(:)));
    global_salc = clean_salc(raw_salc, opts.spsize{i}, -1);
    opts.global_mask = global_salc > opts.beta_global*mean(global_salc(:));

    for pos = 1:numel(opts.peaks)
      idx = opts.peaks(pos);
      [proposals_, root_ids_] = create_proposals(opts, i, idx);
      proposals{i} = [proposals{i}; proposals_];
      root_ids{i} = [root_ids{i}; root_ids_];
    end
    % add the big box covering the image
    proposals{i} = [proposals{i} ; [1,1,opts.imgsize{i}(2),opts.imgsize{i}(1)]];
    [~, max_pixel_idx] = max(global_salc(:));
    root_ids{i} =[root_ids{i}; max_pixel_idx];
    % clean proposals
    [proposals{i}, root_ids{i}] = clean_proposals(proposals{i}, root_ids{i});
    root_feat_types{i} = arrayfun(@(x) opts.feat_type, [1:numel(root_ids{i})]', 'Uni', false);
  end
  
  if opts.save_result
    if exist(opts.save_root, 'file') ~= 7
      mkdir(opts.save_root);
    end
    imdb = struct;
    imdb.bboxes(row_indices,1) = opts.bboxes(row_indices);
    imdb.proposals(row_indices,1) = proposals(row_indices);
    imdb.images_size(row_indices,1) = opts.imgsize(row_indices);
    imdb.root_ids(row_indices,1) = root_ids(row_indices);
    imdb.root_feat_types(row_indices,1) = root_feat_types(row_indices);
    save_path = fullfile(opts.save_root, opts.clname,  ...
                         sprintf('%s_%s_lite', opts.clname, opts.feat_type));
    if exist(save_path, 'file') ~= 7
      mkdir(save_path);
    end
    savefile(fullfile(save_path, sprintf('%d_%d.mat', row_indices(1), row_indices(end))), imdb);
  end
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LOCAL FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [args] = argument_reader(args)
  assert(all(isfield(args, {...
                'root', 'clname', 'small_imdb', 'save_dir', 'feat_type', 'save_result', ...
                'alpha', 'num_maxima', 'ws', 'conn', 'beta_global', 'beta_local', ...
                'num_levels', ...
            })));
  args.save_root = fullfile(args.DATA_ROOT, args.save_dir);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [opts] = get_opts(args, cl)
  opts = struct;
  opts.cl = cl;
  fields = fieldnames(args);
  for i = 1:numel(fields)
    fieldvalues = getfield(args, fields{i});
    opts = setfield(opts, fields{i}, fieldvalues{cl});
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [bboxes, imgsize] = load_imdb(opts)
  if opts.small_imdb
    imdb = load(fullfile(opts.root, opts.clname, [opts.clname, '_small_lite.mat']));
  else
    imdb = load(fullfile(opts.root, opts.clname, [opts.clname, '_lite.mat']));
  end
  bboxes = imdb.bboxes;  
  imgsize = imdb.images_size;
  n = numel(imdb.images_size);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [feat, spsize] = load_feat(opts, i)
  feat = load(fullfile(opts.root, opts.clname, 'features/image', opts.feat_type, sprintf('%d.mat', i)));
  feat = feat.data;
  spsize = [size(feat,2) size(feat,3)];
  feat = reshape(feat, size(feat, 1), [])';
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [salc] = clean_border_topo(salc)
  [h,w] = size(salc);
  h = floor(h/15);
  w = floor(w/15);
  salc(1:h,:) = 0;
  salc(end-h+1:end,:) = 0;
  salc(:,1:w) = 0;
  salc(:,end-w+1:end) = 0;
end

function [salc] = clean_border(salc)
  salc(1,:) = 0;
  salc(end,:) = 0;
  salc(:,1) = 0;
  salc(:,end) = 0;
end

function [salc] = clean_salc_topo(raw_salc, spatial_size, threshold)
  salc = zeros(spatial_size);
  salc(:) = raw_salc;
  salc = clean_border_topo(salc);
  salc = salc .* (salc > threshold*max(salc(:)));
end


function [salc] = clean_salc(raw_salc, spatial_size, threshold)
  salc = zeros(spatial_size);
  salc(:) = raw_salc;
  salc = clean_border(salc);
  salc = salc .* (salc > threshold);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [neighbor_ids] = get_neighbor_ids(idx, h, w)
  neighbor_ids = [idx-1-h idx-1 idx-1+h idx-h idx+h idx+1-h idx+1+h];
  neighbor_ids = neighbor_ids(neighbor_ids > 0);
  neighbor_ids = neighbor_ids(neighbor_ids < w*h);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [local_salc] = build_local_saliency_map(opts, im_idx, peak_idx)
  % opts.feat: (N x d) matrix, l2 normalized CNN features where N is the number of locations.
  
  h = opts.spsize{im_idx}(1);
  w = opts.spsize{im_idx}(2);

  neighbor_ids = get_neighbor_ids(peak_idx, h, w);
  local_salc = zeros(h,w);
  local_salc(:) = opts.feat(peak_idx, :)*opts.feat' + mean(opts.feat(neighbor_ids,:)*opts.feat');
  local_salc = local_salc .* (local_salc > opts.beta_local*mean(local_salc(:)));
  
  % get connected components
  components = bwconncomp(local_salc > 0, opts.conn);
  % keep only clusters whose the highest value >= 1
  % note that scores in the heatmap are in the interval [0,2]
  for cpid = 1:numel(components.PixelIdxList)
    pixel_ids = components.PixelIdxList{cpid};
    if max(local_salc(pixel_ids)) < 1
      local_salc(pixel_ids) = 0;
    end
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [proposals, root_ids] = create_proposals(opts, im_idx, peak_idx)
  sph = opts.spsize{im_idx}(1); spw = opts.spsize{im_idx}(2);
  imgh = opts.imgsize{im_idx}(1); imgw = opts.imgsize{im_idx}(2);

  local_salc = build_local_saliency_map(opts, im_idx, peak_idx);
  local_salc = clean_border(local_salc);
  local_salc = local_salc .* opts.global_mask;

  max_q = nnz(local_salc);
  qlist = round([1:opts.num_levels] * max_q/opts.num_levels);
  [proposals, root_ids] = get_proposals_from_heatmap(local_salc, [imgh, imgw], ...
                                                     qlist, opts.root_threshold, opts.conn);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [proposals, root_ids] = clean_proposals(proposals, root_ids)
  % eliminate duplicating proposals
  [proposals, unique_ids] = unique(proposals, 'rows');
  root_ids = root_ids(unique_ids);
  % eliminate zero-area boxes
  valid_proposal_ids = proposals(:,1) < proposals(:,3) & proposals(:,2) < proposals(:,4);
  valid_proposal_ids = valid_proposal_ids & (root_ids ~= 0);
  proposals = proposals(valid_proposal_ids,:);
  root_ids = root_ids(valid_proposal_ids);
end
