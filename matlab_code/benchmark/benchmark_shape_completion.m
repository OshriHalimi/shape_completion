clearvars; clc; close all; addpath(genpath(fileparts(fileparts(mfilename('fullpath')))));
warning('off', 'manopt:getHessian:approx'); opengl software;
%-------------------------------------------------------------------------%
%                                 Constants
%-------------------------------------------------------------------------%
PATH_SORT_OPT = {'seq','highestL2','lowestL2','rand'};
%-------------------------------------------------------------------------%
%                              Control Panel
%-------------------------------------------------------------------------%
% Directories
c.data_dir =  fullfile(up_script_dir(2),'data');
c.exp_dir = fullfile(c.data_dir,'experiments'); % Presuming same triv
M = read_mesh(fullfile(c.exp_dir,'face_ref.OFF')); c.f = M.f;


% Targets
c.exp_targets = {'EXP16c_Faust2Faust'};
% c.exp_targets = list_file_names(c.exp_dir);
% c.exp_targets = c.exp_targets(startsWith(c.exp_targets,'EXP'));

% Path
c.sort_meth = PATH_SORT_OPT{1};
c.pick_n_at_random = 20;
c.no_self_match = 0;
% c.look_at_sub = {'8','9'};
% c.look_at_template_pose = {'0'};
% c.look_at_projection_pose = {'0'};

% Stat
c.n_run_correspondence = 200;
c.export_correspondence = 1;
c.plot_icp = 0;
c.calc_geodesic_err = 0;

% Visualization
c.n_renders = 0;
c.export_render_to_ply = 0;
c.write_gif = 0; c.frame_rate = 0.3;
c.geodesic_err_xlim = [0,0.2];
%-------------------------------------------------------------------------%
%
%-------------------------------------------------------------------------%

for i=1:length(c.exp_targets)
    c.exp_name = c.exp_targets{i}; c.curr_exp_dir = fullfile(c.exp_dir,c.exp_name);
    fprintf("======================= %s =======================\n",c.exp_name);
    [c,paths] = parse_paths(c);
    stats = collect_reconstruction_err(c,paths);
    [paths,stats] = sort_paths(c,stats,paths);
    [stats] = collect_geoerr(c,stats,paths);
    fprintf('V2V Mean Error : %g cm\n',100*mean(stats.(c.exp_name).ME_err));
    % fprintf('The max L2-Mean-Error is %g\n',max(stats.(c.exp_name).ME_err));
    % fprintf('The min L2-Mean-Error is %g\n',min(stats.(c.exp_name).ME_err));
    
    fprintf('Chamfer GT->Result Error : %g cm\n',100*mean(stats.(c.exp_name).chamfer_gt2res));
    % fprintf('The max Chamfer-Mean-Error is %g\n',max(stats.(c.exp_name).chamfer_err));
    % fprintf('The min Chamfer-Mean-Error is %g\n',min(stats.(c.exp_name).chamfer_err));
    
    fprintf('Chamfer Result->GT Error : %g cm \n',100*mean(stats.(c.exp_name).chamfer_res2gt));
    % fprintf('The max Chamfer2-Mean-Error is %g\n',max(stats.(c.exp_name).chamfer_err2));
    % fprintf('The min Chamfer2-Mean-Error is %g\n',min(stats.(c.exp_name).chamfer_err2));
    
    visualize_statistics(c,stats);
    visualize_results(c,paths); % TODO - refactor this
end
fprintf("======================= DONE =======================\n");
%-------------------------------------------------------------------------%
%                         Central Subroutines
%-------------------------------------------------------------------------%

function [stats] = collect_geoerr(c,stats,paths)

N = min(size(paths,1),c.n_run_correspondence);
geoerr_curves = zeros(N,1001); geoerr_refined_curves = zeros(N,1001);
correspondence = cell(N,1);

if c.export_correspondence
    dset.name = c.exp_name;
    dset.fvs = cell(N+1,4);
    dset.fvs{1,1} = 'Result';
    dset.fvs{1,2} = 'Ground Truth';
    dset.fvs{1,3} = 'Part';
    dset.fvs{1,4} = 'Template';
end

progressbar;
for i=1:N
    [resM,gtM,partM,tempM,mask] = load_path_tup(c,paths(i,:));
    resM.v = compute_icp(resM.v,gtM.v,true);
    moved_part_v = compute_icp(partM.v,resM.v,true);
    
    %====================== ICP TEST ======================%
    if c.plot_icp
        oplt.new_fig=0; oplt.disp_ang = [0,90]; oplt.limits = 0;
        fullfig; subplot_tight(1,2,1);
        
        resM.visualize_vertices(uniclr('teal',resM.Nv),oplt);
        partM.visualize_vertices(uniclr('r',partM.Nv),oplt);
    end
    
    matches_reg = knnsearch(resM.v,partM.v);
    partM.v = moved_part_v;
    
    if c.plot_icp
        subplot_tight(1,2,2);
        resM.visualize_vertices(uniclr('teal',resM.Nv),oplt);
        partM.visualize_vertices(uniclr('r',partM.Nv),oplt);
    end
    %====================== ICP TEST ======================%
    
    correspondence{i} = knnsearch(resM.v,partM.v);
    
    if c.calc_geodesic_err
        D = calc_dist_matrix(tempM.v,tempM.f);
        geoerr_curves(i,:) = calc_geo_err(matches_reg,mask, D);
        geoerr_refined_curves(i,:) = calc_geo_err(correspondence{i},mask, D);
        % matches_refined = refine_correspondence(partM.v,partM.f,tempM.v,tempM.f,matches_reg);
    end
    if c.export_correspondence
        dset.fvs{i+1,1} = resM.fv_struct();
        dset.fvs{i+1,2} = gtM.fv_struct();
        dset.fvs{i+1,3} = partM.fv_struct();
        dset.fvs{i+1,4} = tempM.fv_struct();
    end
    progressbar(i/N);
end
stats.(c.exp_name).geoerr_curves = geoerr_curves;
stats.(c.exp_name).geoerr_refined_curves = geoerr_refined_curves;
stats.(c.exp_name).correspondence = correspondence;
if c.export_correspondence
    dset.correspondence = correspondence;
    save(sprintf('%s_dset.mat',c.exp_name),'dset','-v7.3');
end
end

% [N,I] = conn_comp(partM.A);
% partM = remove_vertices(partM,I~=1);


