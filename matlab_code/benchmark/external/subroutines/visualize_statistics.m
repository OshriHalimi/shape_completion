function visualize_statistics(c,stats)

figure('color','w'); histogram(stats.(c.exp_name).ME_err,30);
title('L2 Error Histogram');
figure('color','w'); histogram(stats.(c.exp_name).chamfer_gt2res,30);
title('Chamfer Ground Truth to Result Error Histogram');
figure('color','w'); histogram(stats.(c.exp_name).chamfer_res2gt,30);
title('Chamfer Result to Ground Truth Error Histogram');

figure('color','w'); hold on; 
plot_geodesic_error(c,stats.(c.exp_name).geoerr_curves);
plot_geodesic_error(c,stats.(c.exp_name).geoerr_refined_curves); 
legend({'Before ICP','After ICP'}); 
hold off; 

end

function plot_geodesic_error(c,curves)
% curves is [N_pairs,1001]
title('Geodesic error (cm)'); 
avg_curve = sum(curves,1)/ size(curves,1);
plot(0:0.001:1.0, avg_curve,'LineWidth',4);
xlim(c.geodesic_err_xlim); ylim([0,1]);
xlabel('Geodesic error (% diameter)')
ylabel('Correspondence Accuracy %'); grid on; 

end