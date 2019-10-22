function [N, nv] = resample(M)
N.m = 0;
while N.m==0 % sometimes remesh() fails
    nv = 500+randi(M.n);
    N = remesh(M, struct('vertices', nv, 'verbose', 0));
end
end
