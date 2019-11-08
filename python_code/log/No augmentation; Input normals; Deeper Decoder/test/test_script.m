addpath(genpath('D:\oshri.halimi\shape_completion\'))
triangulation_path = './smpl_triangulation_source';
full_shape_path = 'D:/oshri.halimi/shape_completion/data/faust_projections/dataset/tr_reg_092.mat';
partial_shape_path = 'D:/oshri.halimi/shape_completion/data/faust_projections/dataset/tr_reg_097_001.mat';
reconstruction_path = './part_tr_reg_097_001_full_tr_reg_092.mat';

full_shape = load(full_shape_path); full_shape = full_shape.full_shape;
partial_shape = load(partial_shape_path); partial_shape = partial_shape.partial_shape;
reconstruction = load(reconstruction_path); 
reconstruction = squeeze(reconstruction.pointsReconstructed)';
x = load(triangulation_path); TRIV = x.TRIV;

trisurf(TRIV,reconstruction(:,1),reconstruction(:,2),reconstruction(:,3)); axis equal
plywrite('./part_tr_reg_097_001_full_tr_reg_092.ply',TRIV,reconstruction)

figure
scatter3(partial_shape(:,1),partial_shape(:,2),partial_shape(:,3),'filled'); axis equal
figure
scatter3(full_shape(:,1),full_shape(:,2),full_shape(:,3),'filled'); axis equal