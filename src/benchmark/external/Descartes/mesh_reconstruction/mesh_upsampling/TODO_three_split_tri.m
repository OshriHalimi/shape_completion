function [MN] = three_split_tri(M,fi)

v1 = M.f(fi,1);
v2 = M.f(fi,2);
v3 = M.f(fi,3);

vn = mean(M.v([v1,v2,v3],:));
vnid = M.Nv +1;

f1 = [v1,v2,vnid]; f2 = [vnid,v2,v3]; f3 = [v1,vnid,v3]; 

M.f(fi,:) = f1;
M.f = [M.f; f2;f3];
M.v = [M.v;vn]; 

MN = Mesh(M.v,M.f,M.name,M.path);

% Sanity: 
% [MN,iters] = mesh_deduplication(MN); 
% assert(iters == 0);
% M3.is_oriented
% MN.singularity([],1);
% MN.boundary([],1);
% assert(MN.is_watertight() && MN.is_2manifold());