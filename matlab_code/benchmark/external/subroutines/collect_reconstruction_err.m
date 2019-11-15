function [stats] = collect_reconstruction_err(c,paths,override)

stats_exists = true;
try
    load('stats.mat','stats');
catch
    stats_exists = false;
end
N = size(paths,1);
if ~stats_exists || ~isfield(stats,c.exp_name)
    
    ME_err = zeros(N,1);
    chamfer_gt2res = zeros(N,1);% gt -> res
    chamfer_res2gt = zeros(N,1); % res -> gt
%     ppm = ParforProgressbar(N);
    progressbar;
    for i=1:N
        [resM,gtM] = load_path_tup(c,paths(i,:));
        resM.v = compute_icp(resM.v,gtM.v,true);
        diff = abs(resM.v - gtM.v).^2; %MSE
        ME_err(i) = sqrt(sum(diff(:))/numel(resM.v));
        [~,D] = knnsearch(resM.v,gtM.v); 
        chamfer_gt2res(i) = mean(D); 
        [~,D] = knnsearch(gtM.v,resM.v); 
        chamfer_res2gt(i) = mean(D); 
        progressbar(i/N); 
%         ppm.increment();
    end
%     delete(ppm);
    stats.(c.exp_name) = struct;
    stats.(c.exp_name).ME_err = ME_err;
    stats.(c.exp_name).chamfer_gt2res = chamfer_gt2res;
    stats.(c.exp_name).chamfer_res2gt = chamfer_res2gt;
    save('stats.mat','stats');
end

fprintf('V2V Mean Error : %g cm\n',100*mean(stats.(c.exp_name).ME_err));
% fprintf('The max L2-Mean-Error is %g\n',max(stats.(c.exp_name).ME_err));
% fprintf('The min L2-Mean-Error is %g\n',min(stats.(c.exp_name).ME_err));

fprintf('Chamfer GT->Result Error : %g cm\n',100*mean(stats.(c.exp_name).chamfer_gt2res));
% fprintf('The max Chamfer-Mean-Error is %g\n',max(stats.(c.exp_name).chamfer_err));
% fprintf('The min Chamfer-Mean-Error is %g\n',min(stats.(c.exp_name).chamfer_err));

fprintf('Chamfer Result->GT Error : %g cm \n',100*mean(stats.(c.exp_name).chamfer_res2gt));
% fprintf('The max Chamfer2-Mean-Error is %g\n',max(stats.(c.exp_name).chamfer_err2));
% fprintf('The min Chamfer2-Mean-Error is %g\n',min(stats.(c.exp_name).chamfer_err2));

end