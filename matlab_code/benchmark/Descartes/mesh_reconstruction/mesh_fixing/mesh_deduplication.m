function [M,iters] = mesh_compaction(M,mode,tol)
if ~exist('mode','var'); mode = 'dec_pts'; end
if ~exist('tol','var'); tol = []; end

% TODO - This function is just sloppy work, fix it. 
Ne =0; Nv = 0; Nf = 0;  
iters = 0;
while(M.Ne ~= Ne && M.Nv ~= Nv && M.Nf ~= Nf)
    Ne = M.Ne; Nv = M.Nv; Nf = M.Nf;
    M = trim_close_vertices(M,mode,tol);
    if Ne ~= M.Ne || Nv ~= M.Nv || Nf ~= M.Nf
        iters = iters + 1;
    end
end
% [vIdx,Vd] = knnsearch(M.v,M.v,'K',2); 
% vIdx
% Vd
