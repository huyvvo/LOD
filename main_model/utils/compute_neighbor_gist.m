
imgset = 'voc';
classes = get_classes(imgset);
num_neighbors = 20;
class_indices = [21:21];
% compute_gist(imgset, class_indices);
compute_neighbors(imgset, class_indices, num_neighbors);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [] = compute_gist(imgset, class_indices)
  % compute fist for image in classes specified by class_indices.
  % the gist features are saved in a separate folder for each class.

  root = ['~/', imgset];
  classes = get_classes(imgset);
  for cl = class_indices
    clname = classes{cl};
    fprintf('Processing for class %s\n', clname);
    imdb = load(fullfile(root, clname, [clname, '_small.mat']));
    n = size(imdb.images, 1);

    gist_path = fullfile(root, 'gist', clname);
    mkdir(gist_path);
    addpath(genpath('/home/vavo/code/ObjectDiscovery_original/tools/gist/'));
    gist_param.orientationsPerScale = [8 8 8 8];
    gist_param.imageSize = [128 128];
    gist_param.numberBlocks = 4;
    gist_param.fc_prefilt = 4;
    nfeat_gist = sum(gist_param.orientationsPerScale)*gist_param.numberBlocks^2;

    fprintf('Processing image ');
    tic;
    for i = 1:n
      fprintf('%d ', i);
      img = imdb.images{i};
      if i == 1
          [gist_feat, gist_param] = LMgist(img, '', gist_param);
      else
          gist_feat = LMgist(img, '', gist_param);
      end
      save(fullfile(gist_path, sprintf('%d.mat', i)), 'gist_feat');
    end
    fprintf('\n');
    fprintf('Took %f secs.\n', toc);
    rmpath(genpath('/home/vavo/code/ObjectDiscovery_original/tools/gist/'));
  end

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [] = compute_neighbors(imgset, class_indices, num_neighbors)
  % Compute a neighborhood for each image in classes specified
  % by class_indices using GIST descriptor.
  % 
  % The neighborhood is the saved to disk.

  root = ['~/', imgset];
  classes = get_classes(imgset);

  addpath(genpath('/home/vavo/code/ObjectDiscovery_original/tools/'));
  
  for cl = class_indices
    clname = classes{cl};
    fprintf('Processing for class %s\n', clname);

    imdb = load(fullfile(root, clname, [clname, '_small.mat']));
    n = size(imdb.images, 1);
    clear imdb;

    neighbor_path = fullfile(root, 'neighbor_gist', clname);
    mkdir(neighbor_path);

    gist_path = fullfile(root, 'gist', clname);  

    % Read GIST features from files
    fprintf('Loading GIST features ...\n');
    gist_feat = cell(n, 1);
    for i = 1:n
        gist_feat{i} = getfield(load(fullfile(gist_path, ...
                                        sprintf('%d.mat', i))), ...
                                'gist_feat');
    end

    % Compute distance between images using GIST features
    gist_feat = cell2mat(gist_feat);
    gist_dist = pwdist_sq(gist_feat', gist_feat');
    gist_dist = gist_dist + diag(ones(n, 1) .* inf);

    fprintf('Computing neighbors ');
    e = cell(n, 1);
    for i = 1:n
      fprintf('%d ', i);
      [~, max_idx] = sort(gist_dist(i,:));
      e{i} = max_idx(1:min(num_neighbors, n-1));
    end
    % save the neighborhood to files
    save(fullfile(neighbor_path, ...
         sprintf('%d.mat', num_neighbors)), 'e');
  end
  rmpath(genpath('/home/vavo/code/ObjectDiscovery_original/tools/'));
end
