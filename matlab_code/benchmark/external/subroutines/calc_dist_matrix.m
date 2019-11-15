function D = calc_dist_matrix(v,f)

nv = size(v,1);
march = fastmarchmex('init', int32(f-1), double(v(:,1)), double(v(:,2)), double(v(:,3)));
D = zeros(nv);

for i=1:nv
    source = inf(nv,1); source(i) = 0;
    d = fastmarchmex('march', march, double(source));
    D(:,i) = d(:);
end

fastmarchmex('deinit', march);
D = 0.5*(D+D');
end