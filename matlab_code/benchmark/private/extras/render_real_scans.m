clearvars; clc; close all; addpath(genpath(fileparts(fileparts(mfilename('fullpath')))));

%-------------------------------------------------------------------------%
%                                 Constants
%-------------------------------------------------------------------------%
path = 'C:\Users\oshri.halimi\Desktop\shake_arm_real_scans';
names = list_file_names(path); 
% load('scan_dfaust_50022_knees.mat','X'); 
load('scan_dfaust_50026_shake_arms.mat','X'); 

X = pointCloud(X(:,1:3)); 
M = read_mesh(fullfile('face_ref.OFF')); f = M.f; oplt.new_fig = 0; 

for i=1:length(names)
    load(fullfile(path,names{i}),'pointsReconstructed');
    pt = pointCloud(squeeze(pointsReconstructed)'); 
    fullfig;
    subplot_tight(1,3,1); 
    pcshow(X,'MarkerSize',25); title('Original'); axis off;  grid off; view([0,90]); 
    subplot_tight(1,3,2); 
    pcshow(pt,'MarkerSize',25); title('Reconstructed'); axis off; grid off; view([0,90]); 
    subplot_tight(1,3,3); 
    recon = Mesh(pt.Location,f,'Reconstructed');view([0,0]); rotate3d; 
    recon.ezvisualize(recon.v(:,3),oplt);
    
end
