function is_allmost = isalmostequal(a,b,tol)
if ~exist('tol','var'); tol = 1e-9; end
is_allmost = all(isalmost(a,b,tol));
end