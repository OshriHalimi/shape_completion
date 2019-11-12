clear all
addpath(genpath('./utils/'))

dir_tosca = 'D:\Data\tosca_downsample_crspnd\shapes\';
dir_save = 'D:\shape_completion\data\tosca_plane_cut\david\';

d = dir([dir_tosca,'david13.mat'])
for i =1:length(d)
    tokens = split(d(i).name,'.');
    name = tokens{1};
    shape = load([d(i).folder,'/',d(i).name]);
    shape.VERT = shape.VERT/100;
    y = shape.VERT(:,2);
    z = shape.VERT(:,3);
    shape.VERT(:,2) = z - 1;
    shape.VERT(:,3) = -y + 0.2;
    
    
    %calculate normal vector
    xyz_triv = [shape.VERT(shape.TRIV(:,1),:), shape.VERT(shape.TRIV(:,2),:),shape.VERT(shape.TRIV(:,3),:)];    
    face_normals = cross(xyz_triv(:,4:6) - xyz_triv(:,1:3),xyz_triv(:,7:9) - xyz_triv(:,4:6));
    norm_face_normals = sum(face_normals.^2,2).^0.5;
    face_normals = face_normals./repmat(norm_face_normals,1,3);
    IVF = I_VF(shape.TRIV,shape.VERT);
    vertex_normals = IVF * face_normals;
    vertex_normals = vertex_normals./repmat(sum(vertex_normals.^2,2).^0.5,1,3);

    shape.x = shape.VERT(:,1);
    shape.y = shape.VERT(:,2);
    shape.z = shape.VERT(:,3);
    
    %save normal vector
    shape.nx = vertex_normals(:,1);
    shape.ny = vertex_normals(:,2);
    shape.nz = vertex_normals(:,3);
    figure; trisurf(shape.TRIV,shape.VERT(:,1),shape.VERT(:,2),shape.VERT(:,3)); axis equal; hold
    quiver3(shape.x,shape.y,shape.z,shape.nx,shape.ny,shape.nz)
    close
    
    points_ind = find(shape.z > -0.05);
    % Duplicate sample points randomly
    N_partial = numel(points_ind);
    rand_ind = randsample(N_partial,shape.n - N_partial,'true');
    points_ind = [points_ind; points_ind(rand_ind)];
    
    full_shape= [shape.x,shape.y,shape.z, ...
                 shape.nx,shape.ny,shape.nz];
    partial_shape = [shape.x(points_ind),shape.y(points_ind),shape.z(points_ind), ...
        shape.nx(points_ind),shape.ny(points_ind),shape.nz(points_ind)];
    
    figure; scatter3(partial_shape(:,1),partial_shape(:,2),partial_shape(:,3)); axis equal; hold
    quiver3(partial_shape(:,1),partial_shape(:,2),partial_shape(:,3),...
        partial_shape(:,4),partial_shape(:,5),partial_shape(:,6))
    close
    
    save([dir_save , name,'.mat'], 'full_shape');
    save([dir_save , name,'_part.mat'], 'partial_shape');
end