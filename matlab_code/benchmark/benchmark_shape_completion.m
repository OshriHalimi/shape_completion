clearvars; close all; clc; addpath(genpath(fileparts(fileparts(mfilename('fullpath')))));
warning('off', 'manopt:getHessian:approx'); opengl software;
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

% Statistics
c.no_self_match = 1;
c.shuffle = 1;
c.look_at_sub = {'9','8'}; 
c.n_stat_collect = 1;
c.n_hist_bins = 20;
c.run_correspondence = 1;
c.geodesic_err_xlim = [0,0.25];

% Visualization
DISP_METH_CHOICES = {'seq','highestL2','lowestL2'};
c.disp_meth = DISP_METH_CHOICES{3};
c.n_renders = 0;
c.write_gif = 1;
c.frame_rate = 0.3;
c.export = 0;
%-------------------------------------------------------------------------%
%
%-------------------------------------------------------------------------%


stats = [];
for i=1:length(c.exp_targets)
    c.exp_name = c.exp_targets{i}; c.curr_exp_dir = fullfile(c.exp_dir,c.exp_name);
    [c,paths] = parse_paths(c);
    [stats] = collect_statistics(c,paths);
    visualize_statistics(c,stats);
    visualize_results(c,paths,stats);
end
%-------------------------------------------------------------------------%
%                         Central Subroutines
%-------------------------------------------------------------------------%

function visualize_results(c,paths,stats) % TODO: Handle more than Faust

if c.export || c.write_gif
    c.exp_res_dir = fullfile(c.exp_dir,sprintf('%s_Visualizations',c.exp_name));
    if ~isfolder(c.exp_res_dir)
        mkdir(c.exp_res_dir)
        fprintf('Created visualizations directiory %s\n',c.exp_res_dir);
    end
end

if c.write_gif
    v = VideoWriter(fullfile( c.exp_res_dir,sprintf('%s_GIF',c.exp_name)));
    v.FrameRate = c.frame_rate; %30 is default
    v.Quality = 100; % 75 is default
    open(v);
    fullfig;
end

switch c.disp_meth
    case 'seq' % Do nothing
    case 'highestL2'
        [~,idx] = sort(stats.MSE_err,'descend');
        paths = paths(idx,:);
    case 'lowestL2'
        [~,idx] = sort(stats.MSE_err,'ascend');
        paths = paths(idx,:);
end


% n_disconnected = 0;
for i=1:min(size(paths,1),c.n_renders)
    
    [resM,gtM,partM,tpM] = load_path_tup(c,paths(i,:),1);
    if ~c.write_gif
        fullfig; % Generate new figure every time
    else
        clf; % Clear the single figure
    end
    
    visualize_meshes(c,gtM,tpM,partM,resM,paths{i,1});
    if c.export
        resM.export_as(fullfile(c.exp_res_dir,sprintf('res_%d.ply',i)));
        gtM.export_as(fullfile(c.exp_res_dir,sprintf('gt_%d.ply',i)));
        partM.export_as(fullfile(c.exp_res_dir,sprintf('part_%d.ply',i)));
        tpM.export_as(fullfile(c.exp_res_dir,sprintf('tp_%d.ply',i)));
    end
    
    if c.write_gif
        writeVideo(v,getframe(gcf));
    end
end

if c.write_gif; close(v); end
end

function [stats] = collect_statistics(c,paths)

% N = size(paths,1);
N = min(size(paths,1),c.n_stat_collect);
MSE_err = zeros(N,1);
if c.run_correspondence
    geoerr_curves = zeros(N,1001); geoerr_refined_curves = zeros(N,1001);
end
% ppm = ParforProgressbar(N);
parfor i=1:N
    [resM,gtM,partM,tempM,mask] = load_path_tup(c,paths(i,:),1);
    
    % COMPUTE MSE
    diff = abs(resM.v - gtM.v).^2;
    MSE_err(i) = sum(diff(:))/numel(resM.v);
    
%     if c.run_correspondence
%         Compute Geodesic Error:
%         
%         matches_reg = knnsearch(resM.v,partM.v);
%         D = calc_dist_matrix(tempM.v,tempM.f);
%         
%         errs = calc_geo_err(matches_reg,mask, D);
%         curve = calc_err_curve(errs, 0:0.001:1.0)/100;
%         geoerr_curves(i,:) = curve;
%         tic;
%         matches_refined = refine_correspondence(partM.v,partM.f,tempM.v,tempM.f,matches_reg,conn_comp(partM.A));
%         toc;
%         errs = calc_geo_err(matches_refined,mask, D);
%         curve = calc_err_curve(errs, 0:0.001:1.0)/100;
%         geoerr_refined_curves(i,:) = curve;
%     end
    %     ppm.increment();
end
% delete(ppm);

% Experiment - Run Geodesic Error over 10 best examples
% N = 10; ppm = ParforProgressbar(N);
if c.run_correspondence
    [~,idx] = sort(MSE_err,'ascend');
    paths = paths(idx,:);
    geoerr_curves = zeros(N,1001); geoerr_refined_curves = zeros(N,1001);
    for i=1:N
        [resM,gtM,partM,tempM,mask] = load_path_tup(c,paths(i,:),1);
        
%         [N,I] = conn_comp(partM.A); 
%         partM = remove_vertices(partM,I~=1); 
        
        
        matches_reg = knnsearch(resM.v,partM.v);
        D = calc_dist_matrix(tempM.v,tempM.f);
        
        errs = calc_geo_err(matches_reg,mask, D);
        curve = calc_err_curve(errs, 0:0.001:1.0)/100;
        geoerr_curves(i,:) = curve;
        tic;
        matches_refined = refine_correspondence(partM.v,partM.f,tempM.v,tempM.f,matches_reg);
        toc;
        errs = calc_geo_err(matches_refined,mask, D);
        curve = calc_err_curve(errs, 0:0.001:1.0)/100;
        geoerr_refined_curves(i,:) = curve;
    end
end
% delete(ppm);

% Place the results in the struct
stats.MSE_err = MSE_err;
if c.run_correspondence
    stats.geoerr_curves = geoerr_curves;
    stats.geoerr_refined_curves = geoerr_refined_curves;
end
fprintf('The L2 MSE mean is %g\n',mean(stats.MSE_err));
fprintf('The max L2 MSE mean is %g\n',max(stats.MSE_err));
fprintf('The min L2 MSE mean is %g\n',min(stats.MSE_err));
end

%-------------------------------------------------------------------------%
%                         Dataset Specific
%-------------------------------------------------------------------------%

% TODO - Insert AMASS/DFAUST case

function [paths] = load_faust_paths(c, outputs)
% {'faust_completion_subjectIDfull_8_poseIDfull_0_subjectIDpart_8_poseIDpart_0_mask_1.mat'}
N = length(outputs); paths = cell(N,4);
for i=1:N
    ids = regexp(outputs{i},'\d*','Match'); % sub,pose,sub,pose,mask
    if isfield(c,'look_at_sub') && ~isempty(c.look_at_sub)
        if ~any(ismember(ids(1),c.look_at_sub))
            continue 
        end
    end
    paths{i,1} = outputs{i}; % The result
    paths{i,2} = sprintf('tr_reg_0%s%s_%03s.mat',ids{3},ids{4},ids{5}); % The part
    paths{i,3} = sprintf('tr_reg_0%s%s.mat',ids{3},ids{4}); % The Ground Truth
    paths{i,4} = sprintf('tr_reg_0%s%s.mat',ids{1},ids{2}); % The Template
    assert(isfile(fullfile(c.ds_dir,paths{i,2})),"Could not find %s",paths{i,2});
    assert(isfile(fullfile(c.ds_dir,paths{i,3})),"Could not find %s",paths{i,3});
    assert(isfile(fullfile(c.ds_dir,paths{i,4})),"Could not find %s",paths{i,4}); % Not really needed
end
end

%-------------------------------------------------------------------------%
%                         Helper Subroutines
%-------------------------------------------------------------------------%

function visualize_statistics(c,stats)

figure('color','w'); histogram(stats.MSE_err,c.n_hist_bins);
title('L2 Error Histogram');
if c.run_correspondence
    plot_geodesic_error(c,stats.geoerr_curves,'Geodesic Error');
    plot_geodesic_error(c,stats.geoerr_refined_curves,'Geodesic Error after Refinement')
end

end

function [c,paths] = parse_paths(c)

outputs = list_file_names(c.curr_exp_dir);
assert(~isempty(outputs),"Directory %s does not contain meshes",c.curr_exp_dir);
% Locate dataset name
[ds_name] = split(outputs{1},'_'); ds_name = ds_name{1};
switch ds_name
    case 'faust'
        c.ds_dir = fullfile(c.data_dir,'faust_projections','dataset');
        %         c.triv_dir = fullfile(c.data_dir,'faust_projections','range_data','res=100x180');
        paths = load_faust_paths(c,outputs);
    otherwise
        error('Unknown dataset %s',ds_name);
end

if c.no_self_match
    goners = false(size(paths,1),1);
    for i=1:size(paths,1) % When the template is exactly the ground truth
        if strcmp(paths{i,3},paths{i,4})
            goners(i) = true;
        end
    end
    paths(goners,:) = [];
end

if c.shuffle
    paths = paths(randperm(size(paths,1)),:);
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
% parttrivfp = fullfile(c.triv_dir,tup_record{2});
gtfp = fullfile(c.ds_dir,tup_record{3});
tpfp = fullfile(c.ds_dir,tup_record{4});

% Loading
gt = load(gtfp); tp = load(tpfp); res = load(resfp); part = load(partfp);
% parttriv = load(parttrivfp);

tpv = double(tp.full_shape(:,1:3)); gtv = double(gt.full_shape(:,1:3));
mask = part.part_mask;
partv =gtv(mask,:);
resv = double(squeeze(res.pointsReconstructed(:,1:3,:)).');
% partv = part.partial_shape(:,1:3);

% Mean Removal - TODO
tpv = tpv - mean(tpv);
gtv = gtv - mean(gtv);
partv = partv - mean(partv);
% resv = resv - mean(resv);


% Mesh Creation
dead_verts = 1:size(gtv,1);
dead_verts(mask) = [];
if as_mesh
    tpv = Mesh(tpv,c.f,'Template');
    gtv = Mesh(gtv,c.f,'Ground Truth');
    partv = remove_vertices(gtv,dead_verts);
    %     partv = Mesh(partv,c.f,'Part');
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


function plot_geodesic_error(c,curves,titl)
% curves is [N_pairs,1001]
avg_curve = sum(curves,1)/ size(curves,1);
figure('color','w');
plot(0:0.001:1.0, avg_curve,'LineWidth',2);
xlim(c.geodesic_err_xlim); ylim([0,1]);
xlabel('Geodesic error')
ylabel('Correspondence Accuracy %')
title(titl)
end


%     Nsegs = conn_comp(partM.A);
%     if Nsegs > 1
%         n_disconnected = n_disconnected +1;
%         fprintf('%d : Num disconnected comp %d\n',n_disconnected,Nsegs);
%         visualize_conncomp(partM)
%     end

