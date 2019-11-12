function [MN,sm,is_mani,is_clsed,is_cmplt,stat] = mesh_status(M)
% First Compact:
[MN,cpc_iters] = mesh_deduplication(M);
is_compact = (cpc_iters==0);

stat = zeros(3,1); stat2 = zeros(3,1);
[is_mani,stat(1)] = manifold_check(MN);
[is_clsed,stat(2)] = watertight_check(MN);
[is_cmplt,stat(3)] = connectivity_check(MN);

if ~is_compact % Sanity Check 
    [is_mani2,stat2(1)] = manifold_check(M);
    [is_clsed2,stat2(2)] = watertight_check(M);
    [is_cmplt2,stat2(3)] = connectivity_check(M);
    % SANITY: Did the deduplication harm the triangulation?
    % Before: Non-singular + Complete + No Boundary
    % After: Singular
    if ~is_mani && is_cmplt2 && is_clsed2 && is_mani2
        warning('Deduplication problem at %s - REVERTING',M.name);
        stat = stat2; is_mani = is_mani2;
        is_clsed = is_clsed2; is_cmplt = is_cmplt2;
        MN = M;
    end
end
%-------------------------------------------------------------------------%
%
%-------------------------------------------------------------------------%
status = {is_compact,is_mani,is_clsed,is_cmplt}; % Add additional checks to here
cstatus = cellfun(@bin2char,status,'UniformOutput',0);
sm = sprintf('%s : Compact[%s] Manifold[%s] Closed[%s] Complete[%s]',M.name,cstatus{:});
if nargout == 0
    fprintf('%s\n',sm); 
end
end
%-------------------------------------------------------------------------%
%
%-------------------------------------------------------------------------%
function [is_mani,stat] = manifold_check(M)
se = M.singularity();
is_mani = isempty(se); stat = size(se,1)./M.Ne;
end
function [is_clsed,stat] = watertight_check(M)
be = M.boundary();
is_clsed = isempty(be); stat = size(be,1)./M.Ne;
end
function [is_cmplt,stat] = connectivity_check(M)
cncomp = conn_comp(M.A);
is_cmplt = (cncomp==1); stat = cncomp;
end