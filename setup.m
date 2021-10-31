LOD_ROOT = pwd();
vlf_path = fullfile(LOD_ROOT,'main_model/han_code/utils/vlfeat-0.9.20/toolbox/vl_setup');
run(vlf_path);

addpath(genpath(fullfile(LOD_ROOT, 'main_model/han_code/utils/commonFunctions')));
addpath(genpath(fullfile(LOD_ROOT, 'main_model/han_code/utils/bbox_tool')));
addpath(genpath(fullfile(LOD_ROOT, 'main_model')));
