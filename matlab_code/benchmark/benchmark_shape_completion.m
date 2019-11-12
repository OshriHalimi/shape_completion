clearvars; close all; clc; addpath(genpath(fileparts(fileparts(mfilename('fullpath')))));opengl software
%-------------------------------------------------------------------------%
%                              Control Panel
%-------------------------------------------------------------------------%

% Directories
c.data_dir =  fullfile(up_script_dir(2),'data');
c.exp_dir = fullfile(c.data_dir,'experiments');
M = read_mesh(fullfile(c.exp_dir,'face_ref.OFF')); % Presuming same triv
c.f = M.f;

% Targets
c.exp_targets = {'EXP16_FaustTrained'};
% Faust Test:
% Exp 17 - Not too good
% Exp 16 - Fantastic
% Exp 13 - Not too good
% Exp 10 - Not too good
% experiment_names = list_file_names(c.exp_dir);

% Visualization
DISP_METH_CHOICES = {'seq','random','highestL2','lowestL2'};

c.no_self_match = 1;
c.n_renders = 10;
c.frame_rate = 0.3;
c.write_gif = 0;
c.disp_meth = DISP_METH_CHOICES{3};

%-------------------------------------------------------------------------%
%
%-------------------------------------------------------------------------%

for i=1:length(c.exp_targets)
    c.exp_name = c.exp_targets{i}; c.curr_exp_dir = fullfile(c.exp_dir,c.exp_name);
    [c,paths] = parse_paths(c);
    [stats] = collect_statistics(c,paths);
    fprintf('The L2 MSE mean is %g\n',mean(stats.MSE_err));
    fprintf('The max L2 MSE mean is %g\n',max(stats.MSE_err));
    fprintf('The min L2 MSE mean is %g\n',min(stats.MSE_err));
    %     load('matlab.mat','stats');
    visualize_results(c,paths,stats);
end
%-------------------------------------------------------------------------%
%                         Central Subroutines
%-------------------------------------------------------------------------%

function visualize_results(c,paths,stats) % TODO: Handle more than Faust

if c.write_gif
    v = VideoWriter(fullfile(c.exp_dir,sprintf('%s_results',c.exp_name)));
    v.FrameRate = c.frame_rate; %30 is default
    v.Quality = 100; % 75 is default
    open(v);
    fullfig;
end

switch c.disp_meth
    case 'seq' % Do nothing
    case 'random'
        paths = paths(randperm(size(paths,1)),:);
    case 'highestL2'
        [~,idx] = sort(stats.MSE_err,'descend');
        paths = paths(idx,:);
    case 'lowestL2'
        [~,idx] = sort(stats.MSE_err,'ascend');
        paths = paths(idx,:);
end

for i=1:min(size(paths,1),c.n_renders)
    
    if c.no_self_match && strcmp(paths{i,3},paths{i,4})
        continue; % When the template is exactly the ground truth
    end
    
    [resM,gtM,partM,tpM] = load_path_tup(c,paths(i,:),1);
    if ~c.write_gif
        fullfig; % Generate new figure every time
    else
        clf; % Clear the single figure
    end
    
    visualize_meshes(c,gtM,tpM,partM,resM,paths{i,1});
    if c.write_gif
        writeVideo(v,getframe(gcf));
    end
end

if c.write_gif; close(v); end
end

function [stats] = collect_statistics(c,paths)

% N = size(paths,1);
N=3;
MSE_err = zeros(N,1);
geoerr_curves = zeros(N,1001);
geoerr_refined_curves = zeros(N,1001);
% Collect L2 Error
% progressbar;
ppm = ParforProgressbar(N);
for i=1:N
    [resv,gtv,partv,tempv,mask] = load_path_tup(c,paths(i,:),0);
    diff = abs(resv - gtv).^2;
    MSE_err(i) = sum(diff(:))/numel(resv); % MSE
    
    % Compute Geodesic Error: 
    D = calc_dist_matrix(tempv,c.f);
    matches_reg = knnsearch(resv,partv); 
    matches_refined = refine_correspondence(partv,resv,matches_reg); 
%     
%     
%     errs = calc_geo_err(matches_reg,mask, D);
%     curve = calc_err_curve(errs, 0:0.001:1.0)/100;
%     geoerr_curves(i,:) = curve;
    
    errs = calc_geo_err(matches_refined,mask, D);
    curve = calc_err_curve(errs, 0:0.001:1.0)/100;
    geoerr_refined_curves(i,:) = curve;
    
    ppm.increment();
    %     progressbar(i/N);
end
delete(ppm);

stats.MSE_err = MSE_err;
stats.geoerr_curves = geoerr_curves;
stats.geoerr_refined_curves = geoerr_refined_curves; 
plot_geodesic_error(geoerr_curves);
plot_geodesic_error(geoerr_refined_curves);

end

%-------------------------------------------------------------------------%
%                         Dataset Specific
%-------------------------------------------------------------------------%

% TODO - Insert AMASS/DFAUST case

function [tups] = load_faust_paths(ds_dir, outputs)
% {'faust_completion_subjectIDfull_8_poseIDfull_0_subjectIDpart_8_poseIDpart_0_mask_1.mat'}
N = length(outputs); tups = cell(N,4);
for i=1:N
    ids = regexp(outputs{i},'\d*','Match'); % sub,pose,sub,pose,mask
    tups{i,1} = outputs{i}; % The result
    tups{i,2} = sprintf('tr_reg_0%s%s_%03s.mat',ids{3},ids{4},ids{5}); % The part
    tups{i,3} = sprintf('tr_reg_0%s%s.mat',ids{3},ids{4}); % The Ground Truth
    tups{i,4} = sprintf('tr_reg_0%s%s.mat',ids{1},ids{2}); % The Template
    
    assert(isfile(fullfile(ds_dir,tups{i,2})),"Could not find %s",tups{i,2});
    assert(isfile(fullfile(ds_dir,tups{i,3})),"Could not find %s",tups{i,3});
    assert(isfile(fullfile(ds_dir,tups{i,4})),"Could not find %s",tups{i,4}); % Not really needed
end
end

%-------------------------------------------------------------------------%
%                         Helper Subroutines
%-------------------------------------------------------------------------%

function [c,tups] = parse_paths(c)

outputs = list_file_names(c.curr_exp_dir);
assert(~isempty(outputs),"Directory %s does not contain meshes",c.curr_exp_dir);
% Locate dataset name
[ds_name] = split(outputs{1},'_'); ds_name = ds_name{1};
switch ds_name
    case 'faust'
        ds_dir = fullfile(c.data_dir,'faust_projections','dataset');
        c.ds_dir = ds_dir;
        tups = load_faust_paths(ds_dir,outputs);
    otherwise
        error('Unknown dataset %s',ds_name);
end
end

function visualize_meshes(c,gtM,tpM,partM,resM,res_name)
suptitle(uplw([c.exp_name,' ',res_name])); axis('off');
oplt.new_fig=0; oplt.disp_ang = [0,90]; oplt.limits = 0;
subplot_tight(1,6,1);
resM.visualize_vertices(uniclr('teal',resM.Nv),oplt);
subplot_tight(1,6,2);
gtM.visualize_vertices(uniclr('teal',gtM.Nv),oplt);
subplot_tight(1,6,3);
resM.visualize_vertices(uniclr('teal',resM.Nv),oplt);
partM.visualize_vertices(uniclr('r',partM.Nv),oplt);
subplot_tight(1,6,4);
partM.visualize_vertices(uniclr('teal',partM.Nv),oplt);
subplot_tight(1,6,5);
tpM.visualize_vertices(uniclr('teal',tpM.Nv),oplt);
subplot_tight(1,6,6);
resM.ezvisualize([],oplt);

% oplt.disp_func = @visualize_vertices;
% mesh_plot_pair(resM,gtM,uniclr('teal',resM.Nv),uniclr('teal',resM.Nv),oplt);
% oplt.disp_func = @ezvisualize;
% mesh_plot_pair(resM,gtM,uniclr('w',resM.Nv),uniclr('w',resM.Nv),oplt);
% mesh_plot_pair(resM,tpM,uniclr('w',resM.Nv),uniclr('w',resM.Nv),oplt);
end


function [resv,gtv,partv,tpv,mask] = load_path_tup(c,tup_record,as_mesh)

% File path creation
resfp =  fullfile(c.curr_exp_dir,tup_record{1});
partfp = fullfile(c.ds_dir,tup_record{2});
gtfp = fullfile(c.ds_dir,tup_record{3});
tpfp = fullfile(c.ds_dir,tup_record{4});

% Loading
gt = load(gtfp); tp = load(tpfp); res = load(resfp); part = load(partfp);
tpv = tp.full_shape(:,1:3); gtv = gt.full_shape(:,1:3);
mask = part.part_mask;
partv =gtv(mask,:);
resv = squeeze(res.pointsReconstructed(:,1:3,:)).';
% partv = part.partial_shape(:,1:3);
% Mean Removal - TODO
tpv = tpv - mean(tpv);
gtv = gtv - mean(gtv);
partv = partv - mean(partv);
% resv = resv - mean(resv);


% Mesh Creation
if as_mesh
    tpv = Mesh(tpv,c.f,'Template');
    gtv = Mesh(gtv,c.f,'Ground Truth');
    partv = Mesh(partv,c.f,'Part');
    resv = Mesh(resv,c.f,'Result');
end

end

function D = calc_dist_matrix(v,f)

nv = size(v,1);
march = fastmarchmex('init', int32(f-1), double(v(:,1)), double(v(:,2)), double(v(:,3)));
D = zeros(nv);

for i=1:nv
    source = inf(nv,1); source(i) = 0;
    d = fastmarchmex('march', march, double(source));
    D(:,i) = d(:);
end

fastmarchmex('deinit', march);
D = 0.5*(D+D');
end


function plot_geodesic_error(curves)
% curves is [N_pairs,1001]
avg_curve = sum(curves,1)/ size(curves,1);
figure('color','w');
plot(0:0.001:1.0, avg_curve,'LineWidth',2);
xlim([0,0.1]); ylim([0,1]);
xlabel('Geodesic error')
ylabel('Correspondence Accuracy %')
title('Geodesic error - all inter pairs')
end

function [matches_upscaled] = refine_correspondence(srcv,tgtv,matches)


src.VERT = srcv; src.n = size(srcv,1);
tgt.VERT = tgtv; tgt.n = size(tgtv,1);


o.k = 100;
o.icp_iters = 10;     % 0 for nearest neighbors
o.use_svd   = true;  % false for basic least squares
o.refine_iters = 25;  % 0 for no refinement

[P,i,j] = create_sparse_matches(src, tgt, src, tgt, matches);
sparse_matches = [i j];
n_matches = size(sparse_matches,1);


% Compute LBO eigenfunctions

[src.W, ~, src.S] = calc_LB_FEM(src);
[src.evecs, src.evals] = eigs(src.W, src.S, o.k, -1e-5);
src.evals = diag(src.evals);
[src.evals, idx] = sort(src.evals);
src.evecs = src.evecs(:,idx);

[tgt.W, ~, tgt.S] = calc_LB_FEM(tgt);
[tgt.evecs, tgt.evals] = eigs(tgt.W, tgt.S, o.k, -1e-5);
tgt.evals = diag(tgt.evals);
[tgt.evals, idx] = sort(tgt.evals);
tgt.evecs = tgt.evecs(:,idx);

% Refine and upscale matches

F = sparse(sparse_matches(:,1), 1:n_matches, 1, src.n, n_matches);
G = sparse(sparse_matches(:,2), 1:n_matches, 1, tgt.n, n_matches);

if o.refine_iters > 0
    
    A_init = src.evecs'*(src.S*F);
    B_init = tgt.evecs'*(tgt.S*G);
    [u,~,v] = svd(A_init*B_init');
    C_init = u*v';
    C_init = C_init';
    
    % fps among the input sparse matches
    fps = fps_euclidean(src.VERT(sparse_matches(:,1),:), 1e3, 1);
    
    matches_upscaled = refine_matches(...
        src, tgt, F(:,fps), G(:,fps), C_init, o);
    
    % do a final svd step
    G_svd = sparse(matches_upscaled, 1:src.n, 1, tgt.n, src.n);
    B_svd = src.evecs'*src.S;
    A_svd = tgt.evecs'*(tgt.S*G_svd);
    [u,~,v] = svd(A_svd*B_svd');
    [~, matches_upscaled] = run_icp_fixed(tgt, src, v*u', o.icp_iters);
    
else
    
    B = src.evecs'*(src.S*F);
    A = tgt.evecs'*(tgt.S*G);
    
    if ~o.use_svd
        C_upscaled = A'\B';
        C_upscaled = C_upscaled';
    else
        [u,~,v] = svd(A*B');
        C_upscaled = u*v';
        C_upscaled = C_upscaled';
    end
    
    [~, matches_upscaled] = run_icp_fixed(tgt, src, C_upscaled, o.icp_iters);
    
end

end

