function [avg_curve,curves] = collect_geoerr(c,stats,paths)

N = min(size(paths,1),c.n_run_geodesic_err);
% if N == 0; return; end

curves = zeros(N,1001);
corr = stats.correspondence;
% progressbar;
if N>0
    ppm = ParforProgressbar(N);
    parfor i=1:N
        
        [~,~,~,tempM,mask] = load_path_tup(c,paths(i,:));
        D = calc_dist_matrix(tempM.v,tempM.f);
        curves(i,:) = calc_geo_err(corr{i},mask, D);
        ppm.increment();
        % progressbar(i/N);
    end
    delete(ppm);
end

avg_curve = sum(curves,1)/ size(curves,1);
end

% matches_refined = refine_correspondence(partM.v,partM.f,tempM.v,tempM.f,matches_reg);