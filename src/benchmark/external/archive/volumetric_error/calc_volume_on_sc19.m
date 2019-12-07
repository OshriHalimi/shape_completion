clear all;close all;clc
% addpath(genpath('./../Utils/'));
addpath(genpath('./../3D_shapes_tools/'));
% addpath(genpath('./../../povray/'));

%% groundtruth volume
gt_volums = zeros(20,1);
gt_surfaces = zeros(20,1);

for pose = 80:99
    filename_in = ['C:\Code\shapecomp19_error\faust_mesh\' sprintf('tr_reg_%.3d.ply', pose)]

    cmnd = ['C:' ' && ' 'cd C:\Program Files\VCG\MeshLab' ' && ' 'meshlabserver -i '...
    filename_in ' -o ' filename_in ' -m fc vc' ' -s ' 'C:\Code\shapecomp19_error\geom_properties.mlx '];
    [ status , cmdout ] = system(cmnd);

    vol_str = 'Mesh Volume  is ';
    i_vol = min(strfind(cmdout, vol_str));
    f_vol = i_vol + strfind(cmdout(i_vol:end), 'LOG') - 3;
    vol = str2num(cmdout(i_vol+numel(vol_str):f_vol));
    
    sur_str = 'Mesh Surface is ';
    i_sur = min(strfind(cmdout, sur_str));
    f_sur = i_sur + strfind(cmdout(i_sur:end), 'LOG') - 3;
    sur = str2num(cmdout(i_sur+numel(sur_str):f_sur));
    
    gt_volums(pose-79) = vol;
    gt_surfaces(pose-79) = sur;
end

figure, plot(gt_volums)
figure, plot(gt_surfaces)

% save('gt_vol','gt_volums','gt_surfaces')

%% sc19 volume
load('EXP16c_Faust_dset.mat')

sc19_volums = zeros(20,10);
sc19_surfaces = zeros(20,10);

for pose = 80:99
    for view = 1:10
        
        % savefile from mat of ply
        out_vert = dset.fvs{1+(pose-80)*10+view,1}.vertices;
        out_tri = dset.fvs{1+(pose-80)*10+view,1}.faces;
        write_ply(out_vert, out_tri, sprintf('recon_mesh_tmp\\sc19_recon_%.3d_%.3d.ply', pose, view))
        
        filename_in = ['C:\Code\shapecomp19_error\recon_mesh_tmp\' sprintf('sc19_recon_%.3d_%.3d.ply', pose, view)]
        
        cmnd = ['C:' ' && ' 'cd C:\Program Files\VCG\MeshLab' ' && ' 'meshlabserver -i '...
        filename_in ' -o ' filename_in ' -m fc vc' ' -s ' 'C:\Code\shapecomp19_error\geom_properties.mlx '];
        [ status , cmdout ] = system(cmnd);
        
        if strfind(cmdout, 'Mesh is not ''watertight''')
            disp('Not watertight. Skipping...')
            vol = 0;
            sur = 0;            
        else

            vol_str = 'Mesh Volume  is ';
            i_vol = min(strfind(cmdout, vol_str));
            f_vol = i_vol + strfind(cmdout(i_vol:end), 'LOG') - 3;
            vol = str2num(cmdout(i_vol+numel(vol_str):f_vol));

            sur_str = 'Mesh Surface is ';
            i_sur = min(strfind(cmdout, sur_str));
            f_sur = i_sur + strfind(cmdout(i_sur:end), 'LOG') - 3;
            sur = str2num(cmdout(i_sur+numel(sur_str):f_sur));
        end

        sc19_volums(pose-79,view) = vol;
        sc19_surfaces(pose-79,view) = sur;
        
    end
end

figure, plot(sc19_volums)
figure, plot(sc19_surfaces)

save('sc19_vol','sc19_volums','sc19_surfaces')

%% analyze sc19 volume
clear all;
load('gt_vol');
load('sc19_vol');
err = [];
for pose = 1:20
    vol_gt = gt_volums(pose);    
    vol_p = sc19_volums(pose,:)';
    vol_p = vol_p(find(vol_p));
    vol_p = abs(vol_p);
    err = [err ; (abs(vol_p - vol_gt) / vol_gt * 100)];
end

mean(err)
std(err,1)


%% poiss volume
poiss_volums = zeros(20,10);
poiss_surfaces = zeros(20,10);

for pose = 80:99
    for view = 1:10
        filename_in = ['C:\Code\shapeCompletion\Poisson\aligned_parts_off\' sprintf('tr_reg_%.3d_%.3d_.off', pose, view)]

        cmnd = ['C:' ' && ' 'cd C:\Program Files\VCG\MeshLab' ' && ' 'meshlabserver -i '...
        filename_in ' -o ' filename_in(1:end-4) '_.off' ' -m fc vc' ' -s ' 'C:\Code\shapeCompletion\geom_properties.mlx '];
        [ status , cmdout ] = system(cmnd);
        
        if strfind(cmdout, 'Mesh is not ''watertight''')
            disp('Not watertight. Skipping...')
            vol = 0;
            sur = 0;            
        else

            vol_str = 'Mesh Volume  is ';
            i_vol = min(strfind(cmdout, vol_str));
            f_vol = i_vol + strfind(cmdout(i_vol:end), 'LOG') - 3;
            vol = str2num(cmdout(i_vol+numel(vol_str):f_vol));

            sur_str = 'Mesh Surface is ';
            i_sur = min(strfind(cmdout, sur_str));
            f_sur = i_sur + strfind(cmdout(i_sur:end), 'LOG') - 3;
            sur = str2num(cmdout(i_sur+numel(sur_str):f_sur));
        end

        poiss_volums(pose-79,view) = vol;
        poiss_surfaces(pose-79,view) = sur;
        
    end
end

figure, plot(poiss_volums)
figure, plot(poiss_surfaces)

save('poiss_vol','poiss_volums','poiss_surfaces')


%% analyze poiss volume
clear all;
load('gt_vol');
load('poiss_vol');
err = [];
for pose = 1:20
    vol_gt = gt_volums(pose);    
    vol_p = poiss_volums(pose,:)';
    vol_p = vol_p(find(vol_p));
    err = [err ; (abs(vol_p - vol_gt) / vol_gt * 100)];
end

mean(err)
std(err,1)

%% voxnet volume
voxnet_volums = zeros(20,10);
voxnet_surfaces = zeros(20,10);

for pose = 80:99
    for view = 1:10
        filename_in = ['C:\Code\shapeCompletion\voxelNet\Results_nfeat32_off\' sprintf('recon_tr_reg_%.3d_%.3d_projdata.off', pose, view)]

        cmnd = ['C:' ' && ' 'cd C:\Program Files\VCG\MeshLab' ' && ' 'meshlabserver -i '...
        filename_in ' -o ' filename_in(1:end-4) '_.off' ' -m fc vc' ' -s ' 'C:\Code\shapeCompletion\flip_normals_and_geom_properties.mlx '];
        [ status , cmdout ] = system(cmnd);
        
        if strfind(cmdout, 'Mesh is not ''watertight''')
            disp('Not watertight. Skipping...')
            vol = 0;
            sur = 0;            
        else

            vol_str = 'Mesh Volume  is ';
            i_vol = min(strfind(cmdout, vol_str));
            f_vol = i_vol + strfind(cmdout(i_vol:end), 'LOG') - 3;
            vol = str2num(cmdout(i_vol+numel(vol_str):f_vol));

            sur_str = 'Mesh Surface is ';
            i_sur = min(strfind(cmdout, sur_str));
            f_sur = i_sur + strfind(cmdout(i_sur:end), 'LOG') - 3;
            sur = str2num(cmdout(i_sur+numel(sur_str):f_sur));
        end

        voxnet_volums(pose-79,view) = vol;
        voxnet_surfaces(pose-79,view) = sur;
        
    end
end

figure, plot(voxnet_volums)
figure, plot(voxnet_surfaces)

save('voxnet_vol','voxnet_volums','voxnet_surfaces')

%% analyze voxnet volume
clear all;
load('gt_vol');
load('voxnet_vol');
err = [];
for pose = 1:20
    vol_gt = gt_volums(pose);    
    vol_p = voxnet_volums(pose,:)';
    vol_p = vol_p(find(vol_p));
    err = [err ; (abs(vol_p - vol_gt) / vol_gt * 100)];
end

mean(err)
std(err,1)

%% ours volume
ours_volums = zeros(20,10);
ours_surfaces = zeros(20,10);

for pose = 80:1:99
    for view = 1:1:10
        filename_in = ['C:\Code\shapeCompletion\Ours\deep_proj_complete_kring2_950_off\' sprintf('tr_reg_%.3d_%.3d.off', pose, view)]

        cmnd = ['C:' ' && ' 'cd C:\Program Files\VCG\MeshLab' ' && ' 'meshlabserver -i '...
        filename_in ' -o ' filename_in(1:end-4) '_.off' ' -m fc vc' ' -s ' 'C:\Code\shapeCompletion\geom_properties.mlx '];
        [ status , cmdout ] = system(cmnd);
        
        if strfind(cmdout, 'Mesh is not ''watertight''')
            disp('Not watertight. Skipping...')
            vol = 0;
            sur = 0;            
        else

            vol_str = 'Mesh Volume  is ';
            i_vol = min(strfind(cmdout, vol_str));
            f_vol = i_vol + strfind(cmdout(i_vol:end), 'LOG') - 3;
            vol = str2num(cmdout(i_vol+numel(vol_str):f_vol));

            sur_str = 'Mesh Surface is ';
            i_sur = min(strfind(cmdout, sur_str));
            f_sur = i_sur + strfind(cmdout(i_sur:end), 'LOG') - 3;
            sur = str2num(cmdout(i_sur+numel(sur_str):f_sur));
        end

        ours_volums(pose-79,view) = vol;
        ours_surfaces(pose-79,view) = sur;
        
    end
end

figure, plot(ours_volums)
figure, plot(ours_surfaces)

save('ours_vol','ours_volums','ours_surfaces')

%% analyze ours volume
clear all;
load('gt_vol');
load('ours_vol');
err = [];
for pose = 1:20
    vol_gt = gt_volums(pose);    
    vol_p = ours_volums(pose,:)';
    vol_p = vol_p(find(vol_p));
    vol_p = abs(vol_p);
    err = [err ; (abs(vol_p - vol_gt) / vol_gt * 100)];
end

mean(err)
std(err,1)

%% ours early termination volume
ours_volums = zeros(20,10);
ours_surfaces = zeros(20,10);

for pose = 80:1:99
    for view = 1:1:10
        filename_in = ['C:\Code\shapeCompletion\Ours\deep_complete_kring2_sparse_iter300only_off\' sprintf('tr_reg_%.3d_%.3d.off', pose, view)]

        cmnd = ['C:' ' && ' 'cd C:\Program Files\VCG\MeshLab' ' && ' 'meshlabserver -i '...
        filename_in ' -o ' filename_in(1:end-4) '_.off' ' -m fc vc' ' -s ' 'C:\Code\shapeCompletion\geom_properties.mlx '];
        [ status , cmdout ] = system(cmnd);
        
        if strfind(cmdout, 'Mesh is not ''watertight''')
            disp('Not watertight. Skipping...')
            vol = 0;
            sur = 0;            
        else

            vol_str = 'Mesh Volume  is ';
            i_vol = min(strfind(cmdout, vol_str));
            f_vol = i_vol + strfind(cmdout(i_vol:end), 'LOG') - 3;
            vol = str2num(cmdout(i_vol+numel(vol_str):f_vol));

            sur_str = 'Mesh Surface is ';
            i_sur = min(strfind(cmdout, sur_str));
            f_sur = i_sur + strfind(cmdout(i_sur:end), 'LOG') - 3;
            sur = str2num(cmdout(i_sur+numel(sur_str):f_sur));
        end

        ours_volums(pose-79,view) = vol;
        ours_surfaces(pose-79,view) = sur;
        
    end
end

figure, plot(ours_volums)
figure, plot(ours_surfaces)

save('ours_300_iter_vol','ours_volums','ours_surfaces')

%% analyze ours early termination volume
clear all;
load('gt_vol');
load('ours_300_iter_vol');
err = [];
for pose = 1:20
    vol_gt = gt_volums(pose);    
    vol_p = ours_volums(pose,:)';
    vol_p = vol_p(find(vol_p));
    vol_p = abs(vol_p);
    err = [err ; (abs(vol_p - vol_gt) / vol_gt * 100)];
end

mean(err)
std(err,1)

%% ours *ORACLE* volume
ours_volums = zeros(20,10);
ours_surfaces = zeros(20,10);

for pose = 80:1:99
    for view = 1:1:10
        filename_in = ['C:\Code\shapeCompletion\Ours\deep_complete_oracle_kring2_best_off\' sprintf('tr_reg_%.3d_%.3d.off', pose, view)]

        cmnd = ['C:' ' && ' 'cd C:\Program Files\VCG\MeshLab' ' && ' 'meshlabserver -i '...
        filename_in ' -o ' filename_in(1:end-4) '_.off' ' -m fc vc' ' -s ' 'C:\Code\shapeCompletion\geom_properties.mlx '];
        [ status , cmdout ] = system(cmnd);
        
        if strfind(cmdout, 'Mesh is not ''watertight''')
            disp('Not watertight. Skipping...')
            vol = 0;
            sur = 0;            
        else

            vol_str = 'Mesh Volume  is ';
            i_vol = min(strfind(cmdout, vol_str));
            f_vol = i_vol + strfind(cmdout(i_vol:end), 'LOG') - 3;
            vol = str2num(cmdout(i_vol+numel(vol_str):f_vol));

            sur_str = 'Mesh Surface is ';
            i_sur = min(strfind(cmdout, sur_str));
            f_sur = i_sur + strfind(cmdout(i_sur:end), 'LOG') - 3;
            sur = str2num(cmdout(i_sur+numel(sur_str):f_sur));
        end

        ours_volums(pose-79,view) = vol;
        ours_surfaces(pose-79,view) = sur;
        
    end
end

figure, plot(ours_volums)
figure, plot(ours_surfaces)

save('ours_oracle_vol','ours_volums','ours_surfaces')

%% analyze ours *ORACLE*  volume
clear all;
load('gt_vol');
load('ours_oracle_vol');
err = [];
for pose = 1:20
    vol_gt = gt_volums(pose);    
    vol_p = ours_volums(pose,:)';
    vol_p = vol_p(find(vol_p));
    vol_p = abs(vol_p);
    err = [err ; (abs(vol_p - vol_gt) / vol_gt * 100)];
end

mean(err)
std(err,1)

%% NN ORACLE volume
ours_volums = zeros(20,10);
ours_surfaces = zeros(20,10);
load('C:\Code\shapeCompletion\NNsearch\error_matrix_100_noICP.mat');
dyfaust_folder = './Data/dfaust_coarse_sparse_set_mat/coarse_sparse_set/';
d = dir(sprintf('%s*.mat',dyfaust_folder));
d = d(1:100:end);

for pose = 80:1:99
    for view = 1:1:10
        
        i_file = best_indices(pose-79, view);
        x = load(sprintf('%s%s',dyfaust_folder, d(i_file).name));    
        N = struct('VERT', x.v, 'TRIV', x.f)'
        
%         filename_in = sprintf('./NNsearch/offs/nn_%.3d_%.3d.off',pose,view);
        filename_in = ['C:\Code\shapeCompletion\NNsearch\offs\' sprintf('tr_reg_%.3d_%.3d.off', pose, view)]

        
        saveoff_color(filename_in ,N.VERT, N.TRIV);
                
        cmnd = ['C:' ' && ' 'cd C:\Program Files\VCG\MeshLab' ' && ' 'meshlabserver -i '...
        filename_in ' -o ' filename_in(1:end-4) '_.off' ' -m fc vc' ' -s ' 'C:\Code\shapeCompletion\geom_properties.mlx '];
        [ status , cmdout ] = system(cmnd);
        
        if strfind(cmdout, 'Mesh is not ''watertight''')
            disp('Not watertight. Skipping...')
            vol = 0;
            sur = 0;            
        else

            vol_str = 'Mesh Volume  is ';
            i_vol = min(strfind(cmdout, vol_str));
            f_vol = i_vol + strfind(cmdout(i_vol:end), 'LOG') - 3;
            vol = str2num(cmdout(i_vol+numel(vol_str):f_vol));

            sur_str = 'Mesh Surface is ';
            i_sur = min(strfind(cmdout, sur_str));
            f_sur = i_sur + strfind(cmdout(i_sur:end), 'LOG') - 3;
            sur = str2num(cmdout(i_sur+numel(sur_str):f_sur));
        end

        nn_volums(pose-79,view) = vol;
        nn_surfaces(pose-79,view) = sur;
        
    end
end

figure, plot(nn_volums)
figure, plot(nn_surfaces)

save('nn_oracle_vol','nn_volums','nn_surfaces')

%% analyze NN ORACLE volume
clear all;
load('gt_vol');
load('nn_oracle_vol');
err = [];
for pose = 1:20
    vol_gt = gt_volums(pose);    
    vol_p = nn_volums(pose,:)';
    vol_p = vol_p(find(vol_p));
    vol_p = abs(vol_p);
    err = [err ; (abs(vol_p - vol_gt) / vol_gt * 100)];
end

mean(err)
std(err,1)
