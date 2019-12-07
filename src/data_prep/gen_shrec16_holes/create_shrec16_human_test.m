clear all
addpath(genpath('./utils/'))

dir_shrec16 = 'D:\Data\SHREC16\shrec2016_PartialDeformableShapes\';
dir_save = 'D:\shape_completion\data\shrec16_evaluation\train_cuts_david\';

d = dir([dir_shrec16,'cuts\cuts_david*.off'])

N_points_full = 10000;
for i = 1:size(d,1)
    tokens = split(d(i).name,'.'); name = tokens{1};
    shape = load_off([d(i).folder,'\',d(i).name]);
    shape.VERT = shape.VERT/100;
    Y_channel = shape.VERT(:,2);
    Z_channel = shape.VERT(:,3);
    shape.VERT(:,2) = Z_channel;
    shape.VERT(:,3) = -Y_channel;
    % Duplicate sample points randomly
    points_ind = 1:shape.n; N_partial = numel(points_ind);
    rand_ind = randsample(N_partial,N_points_full - N_partial,'true');
    points_ind = [points_ind, points_ind(rand_ind)];
    
    
    %calculate normal vector
    xyz_triv = [shape.VERT(shape.TRIV(:,1),:), shape.VERT(shape.TRIV(:,2),:),shape.VERT(shape.TRIV(:,3),:)];    
    face_normals = cross(xyz_triv(:,4:6) - xyz_triv(:,1:3),xyz_triv(:,7:9) - xyz_triv(:,4:6));
    norm_face_normals = sum(face_normals.^2,2).^0.5;
    face_normals = face_normals./repmat(norm_face_normals,1,3);
    IVF = I_VF(shape.TRIV,shape.VERT);
    vertex_normals = IVF * face_normals;
    vertex_normals = vertex_normals./repmat(sum(vertex_normals.^2,2).^0.5,1,3);

    shape.x = shape.VERT(points_ind,1);
    shape.y = shape.VERT(points_ind,2);
    shape.z = shape.VERT(points_ind,3);
    
    %save normal vector
    shape.nx = vertex_normals(points_ind,1);
    shape.ny = vertex_normals(points_ind,2);
    shape.nz = vertex_normals(points_ind,3);
    figure; trisurf(shape.TRIV,shape.VERT(:,1),shape.VERT(:,2),shape.VERT(:,3)); axis equal; hold
    quiver3(shape.x,shape.y,shape.z,shape.nx,shape.ny,shape.nz)
    close
    partial_shape = [shape.x,shape.y,shape.z,shape.nx,shape.ny,shape.nz];
    
    save([dir_save , name, '.mat'], 'partial_shape');
end

d = dir([dir_shrec16,'null\david.off'])
N_points_full = 10000;
for i = 1:size(d,1)
    tokens = split(d(i).name,'.'); name = tokens{1};
    shape = load_off([d(i).folder,'\',d(i).name]);
    shape.VERT = shape.VERT/100;
    Y_channel = shape.VERT(:,2);
    Z_channel = shape.VERT(:,3);
    shape.VERT(:,2) = Z_channel;
    shape.VERT(:,3) = -Y_channel;
    
    % Duplicate sample points randomly
    points_ind = 1:shape.n; N_partial = numel(points_ind);
    rand_ind = randsample(N_partial,N_points_full - N_partial,'true');
    points_ind = [points_ind, points_ind(rand_ind)];
    
    
    %calculate normal vector
    xyz_triv = [shape.VERT(shape.TRIV(:,1),:), shape.VERT(shape.TRIV(:,2),:),shape.VERT(shape.TRIV(:,3),:)];    
    face_normals = cross(xyz_triv(:,4:6) - xyz_triv(:,1:3),xyz_triv(:,7:9) - xyz_triv(:,4:6));
    norm_face_normals = sum(face_normals.^2,2).^0.5;
    face_normals = face_normals./repmat(norm_face_normals,1,3);
    IVF = I_VF(shape.TRIV,shape.VERT);
    vertex_normals = IVF * face_normals;
    vertex_normals = vertex_normals./repmat(sum(vertex_normals.^2,2).^0.5,1,3);

    shape.x = shape.VERT(points_ind,1);
    shape.y = shape.VERT(points_ind,2);
    shape.z = shape.VERT(points_ind,3);
    
    %save normal vector
    shape.nx = vertex_normals(points_ind,1);
    shape.ny = vertex_normals(points_ind,2);
    shape.nz = vertex_normals(points_ind,3);
    figure; trisurf(shape.TRIV,shape.VERT(:,1),shape.VERT(:,2),shape.VERT(:,3)); axis equal; hold
    quiver3(shape.x,shape.y,shape.z,shape.nx,shape.ny,shape.nz)
    close
    full_shape = [shape.x,shape.y,shape.z,shape.nx,shape.ny,shape.nz];
    
    save([dir_save , name, '.mat'], 'full_shape');
end
