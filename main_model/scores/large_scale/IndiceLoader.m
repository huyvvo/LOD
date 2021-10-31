classdef IndiceLoader
%
% Class used to load pairs of image indices (i,j) for score computation.
% Each image pair is given an index which is used to identify it.
%
% Methods
%
% self = IndiceLoader(indice_path, neighbor_path, load_full_indices)
%
% [] = IndiceLoader_write_chunk_symmetric(self, START, END)
%
% [] = IndiceLoader_load_chunk(self, START, END)
%
  properties
    is_symmetric %
    indice_path
    neighbor_path
    indices
  end

  methods

    function self = IndiceLoader(indice_path, neighbor_path, load_full_indices)
      self.indice_path = indice_path;
      self.neighbor_path = neighbor_path;
      self.is_symmetric = true;
      assert(self.is_symmetric ^ contains(self.indice_path, 'symmetric'));
      if load_full_indices
        self.indices = self.load_indices_();
      else 
        self.indices = [];
      end
    end

    function [indices] = load_indices_(self)
      e = load(self.neighbor_path);
      e = e.e;
      n = numel(e);
      
      actual_num_nb = cellfun(@numel, e);
      indices = [repelem([1:n]', actual_num_nb) reshape(cell2mat(e'), [], 1)];
      indices = [indices; indices(:,[2,1])];
      indices = unique(indices, 'rows');
    end

    function [] = IndiceLoader_write_indices(self)
      if isempty(self.indices)
        self.indices = self.load_indices_();
      end
      if exist(self.indice_path,'file') ~= 7
        mkdir(self.indice_path)
      end
      save_data_par(fullfile(self.indice_path, 'indices.mat'), self.indices, 'indices');
    end    

    function [] = IndiceLoader_write_chunk_symmetric(self, START, END)
      if isempty(self.indices)
        self.indices = self.load_indices_();
      end
      END = min(size(self.indices,1),END);
      if START > END
        error('START and END not valid!');
      end
      I = repelem(START:END,2);
      J = repmat(1:2,1,END-START+1);
      V = self.indices(sub2ind(size(self.indices),I,J));

      indices = sparse(I,J,V,size(self.indices,1),2);
      if exist(self.indice_path,'file') ~= 7
        mkdir(self.indice_path)
      end
      save(fullfile(self.indice_path, sprintf('%d_%d.mat', START, END)), 'indices');
    end

    function [indices] = IndiceLoader_load_chunk(self, START, END)
      files = dir(fullfile(self.indice_path, '*_*.mat'));
      files = sort_chunk_name({files.name});
      I = {}; J = {}; V = {};
      SIZE = [];
      for fidx = 1:numel(files)
        fname = files{fidx};
        edges = cellfun(@str2num, strsplit(fname(1:end-4),'_'));
        if edges(2) < START || edges(1) > END 
          continue;
        end
        indices = getfield(load(fullfile(self.indice_path, fname)), 'indices');
        if ~isempty(SIZE)
          assert(isequal(SIZE, size(indices)));
        else 
          SIZE = size(indices);
        end
        [Ic,Jc,Vc] = find(indices);
        I{end+1,1} = Ic;
        J{end+1,1} = Jc;
        V{end+1,1} = Vc;
      end
      I = cell2mat(I);
      J = cell2mat(J);
      V = cell2mat(V);
      indices = sparse(I,J,V,SIZE(1),SIZE(2));
    end
  end % end methods
end % end class