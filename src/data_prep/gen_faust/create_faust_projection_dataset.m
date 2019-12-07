clear all
addpath(genpath('./utils/'))
%dir_faust = 'D:/Data/MPI-FAUST/training/registrations/';
%dir_faust_projections_base = 'D:/Shape-Completion/data/faust_projections/';
dir_range_data = [dir_faust_projections_base , 'range_data/labels/'];
dir_save = [dir_faust_projections_base , 'dataset/'];

d = dir([dir_range_data , '*.mat']);
N_shapes = length(d);
N_points = 6890;

for i=1:N_shapes
    filename = [dir_range_data , d(i).name];
    tokens = split(d(i).name,'_'); shape_idx = tokens{3};
    faust_file_name = [dir_faust , 'tr_reg_' , shape_idx , '.ply'];
    points_ind = load(filename); points_ind = unique(points_ind.labels); N_partial = numel(points_ind);
    rand_ind = randsample(N_partial,N_points - N_partial,'true');
    points_ind = [points_ind; points_ind(rand_ind)];
    
    [mesh,~] = plyread(faust_file_name); 
    full_mesh = [];
    full_mesh.TRIV = cell2mat(mesh.face.vertex_indices) + 1; full_mesh.VERT = [mesh.vertex.x,mesh.vertex.y,mesh.vertex.z];
    
    %calculate normal vector
    xyz_triv = [full_mesh.VERT(full_mesh.TRIV(:,1),:), full_mesh.VERT(full_mesh.TRIV(:,2),:),full_mesh.VERT(full_mesh.TRIV(:,3),:)];    
    face_normals = cross(xyz_triv(:,4:6) - xyz_triv(:,1:3),xyz_triv(:,7:9) - xyz_triv(:,4:6));
    norm_face_normals = sum(face_normals.^2,2).^0.5;
    face_normals = face_normals./repmat(norm_face_normals,1,3);
    IVF = I_VF(full_mesh.TRIV,full_mesh.VERT);
    vertex_normals = IVF * face_normals;
    vertex_normals = vertex_normals./repmat(sum(vertex_normals.^2,2).^0.5,1,3);
    partial_shape = [];
    partial_shape.x = full_mesh.VERT(points_ind,1);
    partial_shape.y = full_mesh.VERT(points_ind,2);
    partial_shape.z = full_mesh.VERT(points_ind,3);
    
    %save normal vector
    partial_shape.nx = vertex_normals(points_ind,1);
    partial_shape.ny = vertex_normals(points_ind,2);
    partial_shape.nz = vertex_normals(points_ind,3);
    %figure; trisurf(full_mesh.TRIV,full_mesh.VERT(:,1),full_mesh.VERT(:,2),full_mesh.VERT(:,3)); axis equal; hold
    %quiver3(full_mesh.VERT(:,1),full_mesh.VERT(:,2),full_mesh.VERT(:,3),vertex_normals(:,1), vertex_normals(:,2),vertex_normals(:,3))
    %close
    full_shape = [full_mesh.VERT(:,1),full_mesh.VERT(:,2),full_mesh.VERT(:,3),vertex_normals(:,1),vertex_normals(:,2),vertex_normals(:,3)];
    partial_shape = [partial_shape.x,partial_shape.y,partial_shape.z, partial_shape.nx, partial_shape.ny, partial_shape.nz];
    
    %full_shape = [full_mesh.VERT(:,1),full_mesh.VERT(:,2),full_mesh.VERT(:,3)];
    %partial_shape = [partial_shape.x,partial_shape.y,partial_shape.z];
    save([dir_save , d(i).name], 'partial_shape');
    save([dir_save , 'tr_reg_' , shape_idx], 'full_shape');
end