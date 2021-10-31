classdef ProposalLoader
%
% Methods
%
% self = ProposalLoader(prop_path, n, prop_step)
%
% [] = ProposalLoader_write_chunk(self, proposals)
%
% [prop] = ProposalLoader_load(self, i)
%
%

  properties
    prop_path % string or char vector, path to proposal folder created by
    n % int, number of images
    prop_step % int, maximum number of proposals in small proposal files in 'prop_paths'
  end

  methods
    function self = ProposalLoader(prop_path, n, prop_step)
    %
    % self = ProposalLoader(prop_path, n, prop_step) 
    %

      self.prop_path = prop_path;
      self.n = n;
      self.prop_step = prop_step;
    
    end

    %------------------------------------

    function [] = ProposalLoader_write_chunk(self, proposals)
    %
    % [] = ProposalLoader_write_chunk(self, proposals) 
    %
      assert(numel(proposals) == self.n);
      assert(exist(self.prop_path) ~= 7);
      mkdir(self.prop_path);

      fprintf('Writing proposal chunks to %s\n', self.prop_path);
      for cidx = 1:ceil(self.n/self.prop_step)
        ctic = tic;
        prop_ids = [(cidx-1)*self.prop_step+1:min(cidx*self.prop_step,self.n)];
        save_struct = struct;
        for idx = prop_ids
          save_struct.(sprintf('proposal_%d',idx)) = proposals{idx};
        end
        savefile(fullfile(self.prop_path, sprintf('%d_%d.mat', min(prop_ids), max(prop_ids))), save_struct);
        fprintf('Chunk %d_%d generated in %.2f secs\n', min(prop_ids), max(prop_ids), toc(ctic));
      end
    end

    function [prop] = ProposalLoader_load(self, i)
    %
    % [prop] = ProposalLoader_load(self, i) 
    %
      k = ceil(i/self.prop_step);
      filename = sprintf('%d_%d.mat', self.prop_step*(k-1)+1, min(self.n,self.prop_step*k));
      prop = load(fullfile(self.prop_path, filename), sprintf('proposal_%d', i));
      prop = prop.(sprintf('proposal_%d',i));
    end

  end % method

end % class