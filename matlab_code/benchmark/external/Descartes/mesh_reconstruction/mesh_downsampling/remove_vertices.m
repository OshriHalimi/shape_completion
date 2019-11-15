function [M_new] = remove_vertices(M, vi)

% create array of indices to keep
newInds = (1:M.Nv)';
newInds(vi) = [];

% create new vertex array
v2 = M.v(newInds, :);

% compute map from old indices to new indices
oldNewMap = zeros(M.Nv, 1);
for iIndex = 1:size(newInds, 1)
   oldNewMap(newInds(iIndex)) = iIndex; 
end

% change labels of vertices referenced by faces
f2 = oldNewMap(M.f);
% keep only faces with valid vertices
f2 = f2(sum(f2 == 0, 2) == 0, :);
M_new = Mesh(v2,f2,M.name,M.path); 


