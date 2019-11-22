clearvars; close all; clc; addpath(genpath(fileparts(fileparts(mfilename('fullpath')))));
warning('off', 'manopt:getHessian:approx'); opengl software;


id = 5;
mesh_prefixes = {'gt','part','res','tp'};

for i=1:length(mesh_prefixes)
    
    M = read_mesh(sprintf('%s_%d.ply',mesh_prefixes{i},id));
    M.plt.disp_ang = [0,90]; M.plt.title= ''; 
    M.ezvisualize();
    material shiny
    saveas(gcf,sprintf('%s_%d.tif',mesh_prefixes{i},id))
end