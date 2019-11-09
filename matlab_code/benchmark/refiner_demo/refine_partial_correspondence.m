%%
% WARNING: challenge data from FAUST challenge contain outlier vertices.
% This demo filters them out and works with cleaned-up meshes; the
% resulting matches must then be mapped back to the original (corrupted)
% raw domains before submitting to the challenge!

clear all
save_dir = 'SHREC16_dog_holes_refined';

addpath('./')
addpath('./tools/')
addpath('./tools/flann/')
addpath(genpath('./Manopt_5.0/manopt/'))

options = struct;
options.k = 100;
options.icp_iters = 10;     % 0 for nearest neighbors
options.use_svd   = true;  % false for basic least squares
options.refine_iters = 25;  % 0 for no refinement


%% Load High Resolution Scans
M = loadoff_colorFix('./shrec_holes/all_shapes/test/holes_dog_shape_4.off'); %M is the source model
N = loadoff_colorFix('./shrec_holes/all_shapes/null/dog.off'); %N is the target model

M.n = size(M.X,1); M.m = size(M.TRIV,1); M.VERT = [M.X, M.Y, M.Z];
N.n = size(N.X,1); N.m = size(N.TRIV,1); N.VERT = [N.X, N.Y, N.Z];
%% Load downsampled FAUST scans

M_dsmpl = M;
N_dsmpl = N;

%% Load sparse matches (these may come from some matching pipeline)
% Warning! sparse_matches matrix dimensions is like the cleaned models and not the
% raw scans!!!

result = load(['./SHREC16_dog_holse_unsupervised_network_results/test_list_000_004.mat']); softCorr = squeeze(result.softCorr);
[~, matches_dsmpl] = max(softCorr,[],1);

[P,i,j] = create_sparse_matches(M_dsmpl, N_dsmpl, M, N, matches_dsmpl);
%[i,j,~] = find(P);
sparse_matches = [i j];
n_matches = size(sparse_matches,1);

colors = create_colormap(N,N);
figure, colormap([1 1 1])
subplot(121)
plot_scalar_map(N, ones(N.n,1)); hold on
plot_cloud_color(N.VERT(sparse_matches(:,2),:), colors(sparse_matches(:,2),:), 5)
axis off; view([0 90])
subplot(122)
plot_scalar_map(M, ones(M.n,1)); hold on
plot_cloud_color(M.VERT(sparse_matches(:,1),:), colors(sparse_matches(:,2),:), 5)
axis off; view([0 90])

%% Compute LBO eigenfunctions

[M.W, ~, M.S] = calc_LB_FEM(M);
[M.evecs, M.evals] = eigs(M.W, M.S, options.k, -1e-5);
M.evals = diag(M.evals);
[M.evals, idx] = sort(M.evals);
M.evecs = M.evecs(:,idx);

[N.W, ~, N.S] = calc_LB_FEM(N);
[N.evecs, N.evals] = eigs(N.W, N.S, options.k, -1e-5);
N.evals = diag(N.evals);
[N.evals, idx] = sort(N.evals);
N.evecs = N.evecs(:,idx);

%% Refine and upscale matches

F = sparse(sparse_matches(:,1), 1:n_matches, 1, M.n, n_matches);
G = sparse(sparse_matches(:,2), 1:n_matches, 1, N.n, n_matches);

if options.refine_iters > 0
    
    A_init = M.evecs'*(M.S*F);
    B_init = N.evecs'*(N.S*G);
    [u,~,v] = svd(A_init*B_init');
    C_init = u*v';
    C_init = C_init';
    
    % fps among the input sparse matches
    fps = fps_euclidean(M.VERT(sparse_matches(:,1),:), 1e3, 1);
    
    matches_upscaled = refine_matches(...
        M, N, F(:,fps), G(:,fps), C_init, options);
    
    % do a final svd step
    G_svd = sparse(matches_upscaled, 1:M.n, 1, N.n, M.n);
    B_svd = M.evecs'*M.S;
    A_svd = N.evecs'*(N.S*G_svd);
    [u,~,v] = svd(A_svd*B_svd');
    [~, matches_upscaled_svd] = run_icp_fixed(N, M, v*u', options.icp_iters);
    
else
    
    B = M.evecs'*(M.S*F);
    A = N.evecs'*(N.S*G);
    
    if ~options.use_svd
        C_upscaled = A'\B';
        C_upscaled = C_upscaled';
    else
        [u,~,v] = svd(A*B');
        C_upscaled = u*v';
        C_upscaled = C_upscaled';
    end
    
    [~, matches_upscaled] = run_icp_fixed(N, M, C_upscaled, options.icp_iters);
    
end

 figure
 colors_hires = create_colormap(N,N);
 subplot(141), colormap(colors_hires), plot_scalar_map(N, 1:N.n); axis off, view([0 90]), freeze_colors
 subplot(142), colormap(colors_hires(matches_dsmpl,:)), plot_scalar_map(M, 1:M.n); axis off, view([0 90]), freeze_colors
subplot(143), colormap(colors_hires), plot_scalar_map(N, 1:N.n); axis off, view([180 -90]), freeze_colors
subplot(144), colormap(colors_hires(matches_dsmpl,:)), plot_scalar_map(M, 1:M.n); axis off, view([180 -90]), freeze_colors

figure
colors_hires = create_colormap(N,N);
subplot(141), colormap(colors_hires), plot_scalar_map(N, 1:N.n); axis off, view([0 90]), freeze_colors
subplot(142), colormap(colors_hires(matches_upscaled,:)), plot_scalar_map(M, 1:M.n); axis off, view([0 90]), freeze_colors
subplot(143), colormap(colors_hires), plot_scalar_map(N, 1:N.n); axis off, view([180 -90]), freeze_colors
subplot(144), colormap(colors_hires(matches_upscaled,:)), plot_scalar_map(M, 1:M.n); axis off, view([180 -90]), freeze_colors


save(['./data/',save_dir,'/',id1,'_',id2,'.mat'],'matches_upscaled')
close all


