function [] = save_data_par(save_path, data, fieldname)
%
% [] = save_data_par(save_path, data, fieldname)
%
  S = struct;
  S = setfield(S, fieldname, data);
  savefile(save_path, S);
end