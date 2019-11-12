function [M_new] = remove_faces(M,fi)
% algorithm
new_inds = (1:M.Nf)';
new_inds(fi) = [];
f2 = M.f(new_inds,:); 
%f2 = M.f(~fi,:);
[unqVertIds, ~, newVertIndices] = unique(f2);
v2 = M.v(unqVertIds,:);
f2 = reshape(newVertIndices,size(f2));
M_new = Mesh(v2,f2,M.name,M.path); 
end

