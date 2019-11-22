function visualize_statistics(c,paths,stats,avg_curve)

if ~c.visualize_stats; return; end

% Histograms:
figure('color','w'); histogram(stats.me_err,30);
title('L2 Error Histogram');

figure('color','w'); histogram(stats.chamfer_gt2res,30);
title('Chamfer Ground Truth to Result Error Histogram');

figure('color','w'); histogram(stats.chamfer_res2gt,30);
title('Chamfer Result to Ground Truth Error Histogram');

% Geodesic Error plot:
plot_geodesic_error(c,avg_curve);
figure('color','w'); set(gcf,'Renderer','painters');
subplot(3,1,1);
plot_azimutal_angle_dep(c,paths,stats.me_err); title('\textbf{Euclidean Distance Vs Projection Angle}','Interpreter','Latex','FontSize',15);
subplot(3,1,2);
plot_azimutal_angle_dep(c,paths,stats.chamfer_gt2res);title('\textbf{Chamfer Distance from GT to Reconstruction Vs Projection Angle}','Interpreter','Latex','FontSize',15);
subplot(3,1,3);
plot_azimutal_angle_dep(c,paths,stats.chamfer_res2gt);title('\textbf{Chamfer Distance from Reconstruction to GT Vs Projection Angle}','Interpreter','Latex','FontSize',15);
% print(gcf,'foo.png','-png','-r1200');
% plot_azimutal_angle_dep(c,paths,stats.vol_err); %title('Volumetric Error with respect to rendering angle','Interpreter','Latex');
end
%-------------------------------------------------------------------------%
%
%-------------------------------------------------------------------------%\
function plot_azimutal_angle_dep(c,paths,err)

err_per_ang = cell(10,1);
for i=1:length(paths)
    ids = paths{i,5};
    mask_id = str2num(ids{5});
    if strcmp(c.curr_tgt_ds,'amass')
        mask_id = mask_id +1 ;
    end
    err_per_ang{mask_id} = [err_per_ang{mask_id},err(i)];
end
stds = 100*cellfun(@std,err_per_ang);
means = 100*cellfun(@mean,err_per_ang);
means = [means ; means(1)] ;
stds = [stds ; stds(1)];
% set(groot,'defaultAxesTickLabelInterpreter','tex');

% barwitherr(stds,means);
% superbar(means,'BarFaceColor', [.3 .3 .9],'E',stds);
errorbar(1:11,means,stds,'-s','LineWidth',2,'MarkerSize',6,'MarkerEdgeColor',[0.6350 0.0780 0.1840],'MarkerFaceColor',[0.6350 0.0780 0.1840]); ylim([0,max(means)+max(stds)+1]);
xticks(1:11);
% xticklabels({'^{0}','^{\pi}/_{5}','^{2\pi}/_{5}','^{3\pi}/_{5}','^{4\pi}/_{5}','\pi','^{6\pi}/_{5}','^{7\pi}/_{5}','^{8\pi}/_{5}','^{9\pi}/_{5}'});
xticklabels(sprintfc('%d^{\\circ}',linspace(0,360,11)));
% xticklabels('0^{\circ},
xlabel('\textbf{Angle}','Interpreter','Latex','FontSize',14);
ylabel('\textbf{Error [cm]}','Interpreter','Latex','FontSize',14);
% set(groot,'defaultAxesTickLabelInterpreter','tex');
end

function  plot_geodesic_error(c,avg_curve)
% curves is [N,1001]

figure('color','w');
plot(0:0.001:1.0, avg_curve,'LineWidth',4);
title('Geodesic error (cm)');
xlim(c.geodesic_err_xlim); ylim([0,1]);
xlabel('Geodesic error (% diameter)')
ylabel('Correspondence Accuracy %'); grid on;
end
% range = 0:0.001:0.2;
% res = avg_curve(1:length(range));
% final = [range.',res.'];