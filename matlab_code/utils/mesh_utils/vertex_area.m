function varea = vertex_area(vertices,faces)
%INPUT: vertices - #vertices X 3 matrix of vertices coordinates: X,Y,Z
%       faces    - #faces X 3 matrix of composing vertices indices for every face
%OUTPUT: varea - per vertex area vector
farea = faces_area(vertices,faces);
adjVF = adjacency_VF(vertices,faces);
varea = (1/3)*adjVF*farea;
end

