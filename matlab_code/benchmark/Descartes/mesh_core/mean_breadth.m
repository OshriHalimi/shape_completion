function mb = mean_breadth(M)
%TRIMESHMEANBREADTH Mean breadth of a triangular mesh
%   Computes the mean breadth (proporitonal to the integral of mean
%   curvature) of a triangular mesh.
%
%   References
%   Stoyan D., Kendall W.S., Mecke J. (1995) "Stochastic Geometry and its
%       Applications", John Wiley and Sons, p. 26
%   Ohser, J., Muescklich, F. (2000) "Statistical Analysis of
%       Microstructures in Materials Sciences", John Wiley and Sons, p.352

v = M.v; f = M.f; 
assert(M.is_2manifold()); 
M = M.add_face_normals(); 
edgeFaceInds = repmat( (1:M.Nf)', 3, 1);
[E,ia,ib] = M.edges(); 

% allocate memory for result
EF = zeros(M.Ne, 2);

% iterate over edges, to identify incident faces
for iEdge = 1:M.Ne
    % Must of 2 length
    EF(iEdge, :) = edgeFaceInds(ib == iEdge);
end


%% Compute dihedral angle for each edge

% compute normal of each face
M = M.add_face_normals(); 

% allocate memory for resulting angles
alpha = zeros(M.Ne, 1);

% iterate over edges
for iEdge = 1:M.Ne
    % indices of adjacent faces
    indFace1 = EF(iEdge, 1);
    indFace2 = EF(iEdge, 2);
    % normal vector of adjacent faces
    normal1 = M.fn(indFace1, :);
    normal2 = M.fn(indFace2, :);
    
    % compute dihedral angle of two vectors
    alpha(iEdge) = vectorAngle3d(normal1, normal2);
end


%% Compute mean breadth
% integrate the dihedral angles weighted by the length of each edge

% compute length of each edge
lengths = meshEdgeLength(v, E);

% compute product of length by angles 
mb = sum(alpha .* lengths) / (4*pi);
