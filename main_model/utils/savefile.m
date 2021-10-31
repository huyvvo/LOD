function [] = savefile(save_path, data)
% SAVEFILE
% [] = savefile(save_path, data)
%
% Save file in the write format (normal or v7.3).
%
% Parameters:
%
%   save_path: string.
%
%   data: struct to save.
%
data_info = whos('data');
if 1000^3*2 < data_info.bytes
  fprintf('Saving file in v7.3 format ...\n');
  save(save_path, '-struct', 'data', '-v7.3');
else 
  fprintf('Saving file in small format ...\n');
  save(save_path, '-struct', 'data');
end

end