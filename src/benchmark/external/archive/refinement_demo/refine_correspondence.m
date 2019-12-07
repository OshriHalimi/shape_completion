function [matches_upscaled] = refine_correspondence(src_v,src_f,tgt_v,tgt_f,matches)

src.VERT = src_v; src.n = size(src_v,1);
src.TRIV = src_f; src.m = size(src_f,1);

tgt.VERT = tgt_v; tgt.n = size(tgt_v,1);
tgt.TRIV = tgt_f; tgt.m = size(tgt_f,1);


o.k = 50;
o.icp_iters = 15;     % 0 for nearest neighbors
o.use_svd   = true;  % false for basic least squares
o.refine_iters = 0;  % 0 for no refinement

[~,i,j] = create_sparse_matches(src, tgt, src, tgt, matches);
sparse_matches = [i j];
n_matches = size(sparse_matches,1);


% Compute LBO eigenfunctions

[src.W, ~, src.S] = calc_LB_FEM(src);
src.S = diag(sqrt(diag(src.S.^2) + 1e-6.^2)); 
[src.evecs, src.evals] = eigs(src.W, src.S, o.k, -1e-5);
src.evals = diag(src.evals);
[src.evals, idx] = sort(src.evals);
src.evecs = src.evecs(:,idx);


[tgt.W, ~, tgt.S] = calc_LB_FEM(tgt);
tgt.S = diag(sqrt(diag(tgt.S.^2) + 1e-6.^2)); 
[tgt.evecs, tgt.evals] = eigs(tgt.W, tgt.S, o.k, -1e-5);
tgt.evals = diag(tgt.evals);
[tgt.evals, idx] = sort(tgt.evals);
tgt.evecs = tgt.evecs(:,idx);

% Refine and upscale matches

F = sparse(sparse_matches(:,1), 1:n_matches, 1, src.n, n_matches);
G = sparse(sparse_matches(:,2), 1:n_matches, 1, tgt.n, n_matches);

if o.refine_iters > 0
    
    A_init = src.evecs'*(src.S*F);
    B_init = tgt.evecs'*(tgt.S*G);
    [u,~,v] = svd(A_init*B_init');
    C_init = u*v';
    C_init = C_init';
    
    % fps among the input sparse matches
    fps = fps_euclidean(src.VERT(sparse_matches(:,1),:), 1e3, 1);
    
    matches_upscaled = refine_matches(...
        src, tgt, F(:,fps), G(:,fps), C_init, o);
    
    % do a final svd step
    G_svd = sparse(matches_upscaled, 1:src.n, 1, tgt.n, src.n);
    B_svd = src.evecs'*src.S;
    A_svd = tgt.evecs'*(tgt.S*G_svd);
    [u,~,v] = svd(A_svd*B_svd');
    [~, matches_upscaled] = run_icp_fixed(tgt, src, v*u', o.icp_iters);
    
else
    
    B = src.evecs'*(src.S*F);
    A = tgt.evecs'*(tgt.S*G);
    
    if ~o.use_svd
        C_upscaled = A'\B';
        C_upscaled = C_upscaled';
    else
        [u,~,v] = svd(A*B');
        C_upscaled = u*v';
        C_upscaled = C_upscaled';
    end
    
    [~, matches_upscaled] = run_icp_fixed(tgt, src, C_upscaled, o.icp_iters);
    
end

end
