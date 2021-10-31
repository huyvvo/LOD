classdef SparseSquareMatrixChunk
%
% This class represents a block of consecutive columns or rows of sparse matrix.
% Since sparse matrices containing many columns are very memory-consuming, 
% blocks of matrix rows and matrix columns are both saved as blocks of matrix columns.
% When the block is a block of rows, its transpose is saved.
% 
% Functions:
%
% self = SparseSquareMatrixChunk(I, J, V, N, is_row, is_symmetric, first_index, second_index)
% res = SSMC_sum_by_row(self)
% res = SSMC_sum_by_col(self)
% res = SSMC_left_mul(self,v)
% res = SSMC_right_mul(self,v)
% res = SSMC_get_indices(self
%


  properties
    first_index % float, index of the first row or column in the block
    second_index % int, index of the last row or column in the block
    is_row % boolean, whether the original form of the block is row-based
    is_symmetric % boolean, whether the matrix is symmetric
    N % int, dimension of rows in row-based blocks or columns in column-based blocks
    S % sparse matrix of size (N x (second_index-first_index+1))
  end

  methods
    function self = SparseSquareMatrixChunk(I, J, V, N, is_row, is_symmetric, first_index, second_index)
      % SparseSquareMatrixChunk(J, I, V, N, is_row, first_index, second_index)
      % 
      % Parameters:
      % I: row indices of non-zero elements
      % J: column indices of non-zero elements
      % V: values of non-zero elements
      % N: matrix size
      % is_row: whether the chunk is row-based or column-based
      % first_index
      % second_index

      if is_row
        assert(min(I) >= first_index && max(I) <= second_index);
      else 
        assert(min(J) >= first_index && max(J) <= second_index);
      end
      
      self.first_index = first_index;
      self.second_index = second_index;
      self.is_row = is_row;
      self.is_symmetric = is_symmetric;
      self.N = N;
      if is_row
        self.S = sparse(J,I-first_index+1,V,N,second_index-first_index+1);
      else
        self.S = sparse(I,J-first_index+1,V,N,second_index-first_index+1); 
      end
    end

    %------------------------------------

    function res = SSMC_sum_by_row(self)
      if self.is_row
        res = sum(self.S,1)';
      else 
        error('sum_by_row for column chunks is not implemented!');
      end
    end

    function res = SSMC_sum_by_col(self)
      if self.is_row
        error('sum_by_col for row chunks is not implemented!');
      else
        res = sum(self.S,1); 
      end
    end

    function res = SSMC_left_mul(self,v)
      if self.is_row
        if self.is_symmetric
          res = v*self.S;
        else 
          error('Left multiplication for row chunks is not implemented!');
        end
      else
        res = v*self.S;
      end
    end

    function res = SSMC_right_mul(self,v)
      if self.is_row
        res = (v'*self.S)';
      else 
        error('Left multiplication for row chunks is not implemented!');
      end
    end  

    function res = SSMC_get_indices(self)
      res = [self.first_index self.second_index];
    end  

  end % method

end % class