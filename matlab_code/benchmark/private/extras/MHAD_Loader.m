clearvars; clc ; close all;
%=========================================================================%
%
%=========================================================================%
TP_DIR = 'C:\Users\idoim\Desktop\chicken_wings';
ORIG_DIR = 'C:\Users\idoim\Desktop\Real Scans';
RES_DIR = 'C:\Users\idoim\Desktop\scans_mhad_chicken_wings_50002';
N_SCANS = 3;
%=========================================================================%
%
%=========================================================================%
for i=1:N_SCANS
    
    tgt_real_scan_name  = sprintf('mhad_real_scan_%d',i);
    orig_scan = load(fullfile(ORIG_DIR, [tgt_real_scan_name ,'.mat']));
    % orig_scan.xyz
    [~,result_files] = list_file_names(fullfile(RES_DIR,tgt_real_scan_name));
    n_res = numel(result_files);
    for j=0:10:n_res-1
        % Get Template
        tp_mesh = read_mesh(fullfile(TP_DIR,sprintf('%05d.OFF',j)));
        % Get Result
        res = load(result_files{j+1}); % Presuming sorted order - Weak presumption
        
        fullfig;
        
        subplot_tight(1,3,1);
        scatter3(orig_scan.xyz(:,1),orig_scan.xyz(:,2),orig_scan.xyz(:,3),10,orig_scan.xyz(:,2),'filled');
%         scatter3sph(orig_scan.xyz(:,1),orig_scan.xyz(:,2),orig_scan.xyz(:,3),'size',3);
        view([0,90]); axis off;
        
        subplot_tight(1,3,2);
        opt.disp_ang = [0,90]; opt.new_fig = 0; 
        res_xyz = squeeze(res.pointsReconstructed).';
        res_mesh = Mesh(res_xyz,tp_mesh.f,'Results');
        res_mesh.ezvisualize([],opt);
        
        subplot_tight(1,3,3);
        tp_mesh.ezvisualize([],opt); 
        
    end 
end