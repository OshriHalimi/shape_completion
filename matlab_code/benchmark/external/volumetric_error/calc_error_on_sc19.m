clear all;close all;clc
addpath(genpath('./../3D_shapes_tools/'));
addpath(genpath('./../gptoolbox-master'));
%% params
% 
% res_folder = 'deep_complete_kring2_sparse_iter300only';
% res_folder = 'deep_complete_oracle_kring2_best';
% res_folder = 'deep_proj_complete_kring2_950';
% res_folder = 'shape_complete_nores_all_M8_L1e-8_me_D128_monetsmooth_allposes';
% res_folder = 'shape_complete_nores_ho_M8_L1e-8_fc_D64_monetsmooth';
% res_folder = 'shape_complete_nores_all_M8_L1e-8_me_D128_monetsmooth';
% res_folder = 'shape_complete_nores_ho_M8_L1e-6_fc_D64_monetsmooth';
% res_folder = 'shape_complete_nores_all_M8_L1e-6_me_D128_monetsmooth';
% res_folder = 'nores_all_M8_L1e-6_me_D128';
% res_folder = 'nores_ho_M8_L1e-6_fc_D64';
%% Load results from sc19
load('EXP16c_Faust_dset.mat')

%% calc errors on unseen part
err_matrix = zeros(20,10);

for pose = 80:1:99
    gt_fname = sprintf('./faust_mesh/tr_reg_%.3d.ply', pose);
    % load gt full shape
    [S.VERT, S.TRIV] = read_ply(gt_fname)
    S.n = size(S.VERT, 1);
    
    for view = 1:1:10        
        out_vert = dset.fvs{1+(pose-80)*10+view}.vertices;
        [~, D] = knnsearch(out_vert ,S.VERT);

        err_matrix(pose - 79, view) = mean(D)*100;
    end
end

save('sc19_err_matrix','err_matrix');
%% save figure
load('errors_kring2_best.mat')
fig = figure; imagesc(1:2:10, 80:2:99, err_matrix); ylabel('pose'); xlabel('view'); colorbar; caxis([0 6]);
saveas(fig, 'error_matrix_ours.png');

%%
pose = 85
for view = 5    
    out_vert = dset.fvs{1+(pose-80)*10+view,1}.vertices;
    out_tri = dset.fvs{1+(pose-80)*10+view,1}.faces;
    figure,plot_mesh(out_vert, out_tri)
end
gt_fname = sprintf('./faust_mesh/tr_reg_%.3d.ply', pose);
% load gt full shape
[S.VERT, S.TRIV] = read_ply(gt_fname)
plot_mesh(S.VERT, S.TRIV)


