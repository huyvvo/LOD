function sig = reconstruct_sig(nx,ny,ww,dxy)
% RECONTRUCT_SIG

% sig = reconstruct_sig(nx,ny,ww,dxy)
% 
% Parameters:
%
%   nx: int, number of columns of the windows.
%
%   ny: int, number of rows of the windows.
%
%   ww: (d x d x num_offset),
%
%   dxy: list of offsets
%
% Returns:
%
% 

  k  = size(dxy,1);
  nf = size(ww,1);
  n  = ny*nx;  
  sig  = zeros(nf,nf,n,n);
  
  for x1 = 1:nx,
    for y1 = 1:ny,
      i1 = (x1-1)*ny + y1;
      for i = 1:k,
        x = dxy(i,1);
        y = dxy(i,2);
        x2 = x1 + x;        
        y2 = y1 + y;
        if x2 >= 1 && x2 <= nx && y2 >= 1 && y2 <= ny,
          i2 = (x2-1)*ny + y2;
          sig(:,:,i1,i2) = ww(:,:,i); 
        end
        x2 = x1 - x;        
        y2 = y1 - y;
        if x2 >= 1 && x2 <= nx && y2 >= 1 && y2 <= ny,
          i2 = (x2-1)*ny + y2; 
          sig(:,:,i1,i2) = ww(:,:,i)'; 
        end
      end
    end
  end
  
  % Permute [nf nf n n] to [n nf n nf]
  sig = permute(sig,[3 1 4 2]);
  sig = reshape(sig,n*nf,n*nf);
  
  % Make sure returned matrix is close to symmetric
  assert(sum(sum(abs(sig - sig'))) < 1e-5);
  
  sig = (sig+sig')/2;
