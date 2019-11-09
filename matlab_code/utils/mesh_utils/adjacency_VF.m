function adjVF = adjacency_VF(vertices,faces)
%INPUT:  vertices - #vertices X 3 matrix of vertices coordinates: X,Y,Z
%        faces - #faces X 3 matrix of composing vertices indices for every face
%OUTPUT: adjVF - adjacency matrix, the ij-entry is 1 if vertex i is
%neighbor of face j, else 0
N_vertices = size(vertices,1);
N_faces = size(faces,1);

adjVF = sparse([faces(:,1); faces(:,2); faces(:,3)],repmat(1:N_faces,1,3),ones(3*N_faces,1),N_vertices,N_faces);
end

