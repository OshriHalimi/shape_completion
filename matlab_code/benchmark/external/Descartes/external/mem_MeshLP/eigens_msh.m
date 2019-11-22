%opt.dtype='geodesic';
opt.dtype='euclidean';
opt.htype='ddr';
opt.hs = 2;
opt.rho = 3;
nev = 300;


prefix = 'bighand';
prefix = 'object'
filename=sprintf('./%s.off', prefix);
[W A h] = symmshlp_matrix(filename, opt);
Am = sparse([1:length(A)], [1:length(A)], A);
[evecs evals] = eigs(W, Am, nev, -1e-5);
evals = diag(evals);

filename=sprintf('./%s_sym_dt_%s_ht_%s_hs%d_rho%d', prefix, opt.dtype, opt.htype, opt.hs, opt.rho);
save(filename, 'W', 'A', 'evecs', 'evals', 'h');
