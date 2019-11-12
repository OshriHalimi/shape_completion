clearvars; close all; clc; addpath(genpath(fileparts(fileparts(mfilename('fullpath')))));opengl software
%-------------------------------------------------------------------------%
%                              Control Panel
%-------------------------------------------------------------------------%

% Directories
c.data_dir =  fullfile(up_script_dir(2),'data');
c.exp_dir = fullfile(c.data_dir,'experiments');
M = read_mesh(fullfile(c.exp_dir,'face_ref.OFF')); % Presuming same triv
c.F = M.f;

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

N = size(paths,1);
MSE_err = zeros(N,1);
% Collect L2 Error
progressbar;
for i=1:N
    [resv,gtv,partv,tpv] = load_path_tup(c,paths(i,:),0);
    D = abs(resv - gtv).^2;
    MSE_err(i) = sum(D(:))/numel(resv); % MSE 
    progressbar(i/N);
    
end


stats.MSE_err = MSE_err;

%     idx = knnsearch(partM.v,resM.v)

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


function [resv,gtv,partv,tpv] = load_path_tup(c,tup_record,as_mesh)

% File path creation
resfp =  fullfile(c.curr_exp_dir,tup_record{1});
partfp = fullfile(c.ds_dir,tup_record{2});
gtfp = fullfile(c.ds_dir,tup_record{3});
tpfp = fullfile(c.ds_dir,tup_record{4});

% Loading
gt = load(gtfp); tp = load(tpfp); res = load(resfp); part = load(partfp);
tpv = tp.full_shape(:,1:3); gtv = gt.full_shape(:,1:3);
partv = part.partial_shape(:,1:3); resv = squeeze(res.pointsReconstructed(:,1:3,:)).';

% Mean Removal - TODO
tpv = tpv - mean(tpv);
gtv = gtv - mean(gtv);
partv = partv - mean(partv);
% resv = resv - mean(resv);


% Mesh Creation
if as_mesh
    tpv = Mesh(tpv,c.F,'Template');
    gtv = Mesh(gtv,c.F,'Ground Truth');
    partv = Mesh(partv,c.F,'Part');
    resv = Mesh(resv,c.F,'Result');
end

end



