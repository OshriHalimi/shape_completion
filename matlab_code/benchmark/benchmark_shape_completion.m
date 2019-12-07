clearvars; clc; close all; addpath(genpath(fileparts(fileparts(mfilename('fullpath')))));cd(up_script_dir(0)); 
opengl software;
%-------------------------------------------------------------------------%
%                                 Constants
%-------------------------------------------------------------------------%
PATH_SORT_OPT = {'seq','highestL2','lowestL2','lowest_chamfergt2res','lowest_chamferres2gt','rand','none'};
c.path.data_dir =  fullfile(up_script_dir(2),'data');
c.path.exps_dir = fullfile(c.path.data_dir,'experiments');
c.path.collat_dir = fullfile(up_script_dir(0),'collaterals');
c.path.tmp_dir = fullfile(c.path.collat_dir,'tmp');
[c.f,c.f_ds] = get_representative_tri(c); % Presuming same triv
%-------------------------------------------------------------------------%
%                              Control Panel
%-------------------------------------------------------------------------%

% Targets
% c.exp_targets = {'EXP16c_Faust2Faust'};
% c.exp_targets = {'EXP_Ablation_2Faust'};
c.exp_targets = {'EXP21_Amass2Amass'};
% c.exp_targets = list_file_names(c.path.exps_dir);
% c.exp_targets = c.exp_targets(startsWith(c.exp_targets,'EXP'));

% Path
c.sort_meth = PATH_SORT_OPT{4};
% c.pick_n_at_random = 20;
c.no_self_match = 1;
% c.look_at_sub = {'8','9'};
% c.look_at_template_pose = {'0'};
% c.look_at_projection_pose = {'0'};
% c.look_at_projection_id = {'6','3','8'};
c.export_subset = 0;

% Geodesic Error
c.n_run_geodesic_err = 0;
c.geodesic_err_xlim = [0,0.2];
c.visualize_stats = 0;

% Visualization
c.n_renders = 30;
c.cherry_pick_mode = 1;
c.export_render_to_ply = 1;
c.write_gif = 0; c.frame_rate = 0.3;

%-------------------------------------------------------------------------%
%
%-------------------------------------------------------------------------%

for i=1:length(c.exp_targets)
    banner(c.exp_targets{i});
    c = tailor_config_to_exp(c,c.exp_targets{i});
    paths = parse_paths(c);
    [stats,~] = collect_reconstruction_stats(c,paths);
    [paths,stats] = filter_subset(c,stats,paths);
    [avg_curve,~] = collect_geoerr(c,stats,paths);
    visualize_statistics(c,paths,stats,avg_curve);
    visualize_results(c,stats,paths);
end
banner('done');



