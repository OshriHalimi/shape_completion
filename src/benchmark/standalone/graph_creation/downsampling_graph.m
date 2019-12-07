clearvars; clc; close all;
% Parse the noise experiment names:
load('downsampling_cache.mat','stats'); 
fnames = fieldnames(stats);

N = length(fnames); % Assuming downsampling experiments are the only fields
X = zeros(N,1);
me_means = zeros(N,1); me_stds = zeros(N,1);
gt2res_means = zeros(N,1); gt2res_stds = zeros(N,1);
res2gt_means = zeros(N,1); res2gt_stds = zeros(N,1);

for i=1:N
    noise_ids = regexp(fnames{i},'\d*','Match');
    assert(length(noise_ids)==3);
    noise_id = [ '0.' noise_ids{2}];
    noise_id = str2num(noise_id);
    X(i) = 100*noise_id;
    me_means(i) = 100*mean(stats.(fnames{i}).me_err);
    me_stds(i) = 100*std(stats.(fnames{i}).me_err);
    
    gt2res_means(i) = 100*mean(stats.(fnames{i}).chamfer_gt2res);
    gt2res_stds(i) = 100*std(stats.(fnames{i}).chamfer_gt2res);
    
    res2gt_means(i) = 100*mean(stats.(fnames{i}).chamfer_res2gt);
    res2gt_stds(i) = 100*std(stats.(fnames{i}).chamfer_res2gt);
end

[X,idx] = sort(X,'ascend'); 
me_means = me_means(idx); me_stds = me_stds(idx); 
res2gt_means = res2gt_means(idx); gt2res_stds = gt2res_stds(idx); 
res2gt_means = res2gt_means(idx); res2gt_stds = res2gt_stds(idx); 

figure('color','w','Renderer','painters');
subplot(3,1,1); 
plot_graph(X,me_means,me_stds);title('\textbf{Euclidean Distance Vs Downsampling Precentage}','Interpreter','Latex','FontSize',15); 
% print(gcf,'noise1.png','-dpng','-r1200');
subplot(3,1,2); 
plot_graph(X,gt2res_means,gt2res_stds); title('\textbf{GT to Reconstruction Vs Downsampling Precentage}','Interpreter','Latex','FontSize',15); 
% print(gcf,'noise2.png','-dpng','-r1200');
subplot(3,1,3); 
plot_graph(X,res2gt_means,res2gt_stds); title('\textbf{Reconstruction to GT Vs Downsampling Precentage}','Interpreter','Latex','FontSize',15); 
% print(gcf,'noise_analysis.png','-dpng','-r1200');

function plot_graph(X,means,stds)

errorbar(X,means,stds,'-s','LineWidth',2,'MarkerSize',6,'MarkerEdgeColor',[0.3350 0.3780 0.6840],'MarkerFaceColor',[0.6350 0.0780 0.1840]); 
% [0.4940 0.1840 0.5560]
ylim([mean(means)-mean(stds)-1,mean(means)+mean(stds)+1]); 
% ylim
hold on;
plot(X,means(1)*ones(length(X),1),'r--','LineWidth',1); 

% xticklabels(sprintfc('%d^{\\circ}',linspace(0,360,11)));
xlabel('\textbf{Precentage of Missing Vertices [\%]}','Interpreter','Latex','FontSize',14);
ylabel('\textbf{Error [cm]}','Interpreter','Latex','FontSize',14);
end




