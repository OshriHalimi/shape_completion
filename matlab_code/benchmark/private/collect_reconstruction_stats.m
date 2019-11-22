function [curr_stats,stats] = collect_reconstruction_stats(c,paths,force)

banner('Reconstruction Error Report'); 
if ~exist('force','var'); force = 0; end
tgt_stats_fp = fullfile(c.path.collat_dir,'stats_cache.mat');

% Check for the existence of the stats cache
try
    load(tgt_stats_fp,'stats');
    stats_exists = true;
catch
    stats_exists = false;
end

% If it is invalid for the current experiment - compute and save
if force || ~stats_exists || ~isfield(stats,c.curr_exp)
    
    N = size(paths,1);
    me_err = zeros(N,1); vol_err = zeros(N,1);
    chamfer_gt2res = zeros(N,1); chamfer_res2gt = zeros(N,1);
    correspondence = cell(N,1); correspondence_10_hitrate = zeros(N,1);
    ppm = ParforProgressbar(N);
    %     progressbar;
    parfor i=1:N
        [resM,gtM,partM,~,mask] = load_path_tup(c,paths(i,:));
        if resM.Nf ~= gtM.Nf
            gtM = qslim(gtM,resM.Nf);
        end
        % Align Result<->GT
        resM.v = compute_icp(resM.v,gtM.v,true);
        
        % Compute Mean Error
        diff = abs(resM.v - gtM.v).^2; %MSE
        me_err(i) = sqrt(sum(diff(:))/numel(resM.v));
        % Compute Volume Error
        gtvol = gtM.volume();
        vol_err(i) = abs(gtvol - resM.volume())/gtvol;
        % Compute Chamfer
        [~,D] = knnsearch(resM.v,gtM.v);
        chamfer_gt2res(i) = mean(D);
        [~,D] = knnsearch(gtM.v,resM.v);
        chamfer_res2gt(i) = mean(D);
        % Align Part<->Res
        % matches_reg = knnsearch(resM.v,partM.v); % Before move
        partM.v = compute_icp(partM.v,resM.v,true);
        correspondence{i} = knnsearch(resM.v,partM.v);
        correspondence_10_hitrate(i) = nnz(~(correspondence{i}-mask))/numel(mask);
        
        %         progressbar(i/N);
        ppm.increment();
    end
    delete(ppm);
    stats.(c.curr_exp) = struct;
    stats.(c.curr_exp).me_err = me_err;
    stats.(c.curr_exp).chamfer_gt2res = chamfer_gt2res;
    stats.(c.curr_exp).chamfer_res2gt = chamfer_res2gt;
    stats.(c.curr_exp).vol_err = vol_err;
    stats.(c.curr_exp).correspondence = correspondence;
    stats.(c.curr_exp).correspondence_10_hitrate = correspondence_10_hitrate;
    save(tgt_stats_fp,'stats');
end

% Report the current statistics
curr_stats = stats.(c.curr_exp);
report_moments('Euclidean V2V',curr_stats.me_err,'cm');
report_moments('Volume',curr_stats.vol_err,'%');
report_moments('Chamfer GT->Result',curr_stats.chamfer_gt2res,'cm');
report_moments('Chamfer Result->GT',curr_stats.chamfer_res2gt,'cm');
report_moments('Correspondence Direct Hits',curr_stats.correspondence_10_hitrate,'%');
end




% Hacky way to resolve vertex misalignment
%         try
%         diff = abs(resM.v - gtM.v).^2; %MSE
%         catch
%             gtM = qslim(gtM,resM.Nf);
%             Nv = min(resM.Nv,gtM.Nv);
%             diff = abs(resM.v(1:Nv,:) - gtM.v(1:Nv,:)).^2; %MSE
%         end
%     ====================== ICP TEST ======================%
%     if c.plot_icp
%         oplt.new_fig=0; oplt.disp_ang = [0,90]; oplt.limits = 0;
%         fullfig; subplot_tight(1,2,1);
%
%         resM.visualize_vertices(uniclr('teal',resM.Nv),oplt);
%         partM.visualize_vertices(uniclr('r',partM.Nv),oplt);
%     end
%
%     matches_reg = knnsearch(resM.v,partM.v);
%     partM.v = moved_part_v;
%
%     if c.plot_icp
%         subplot_tight(1,2,2);
%         resM.visualize_vertices(uniclr('teal',resM.Nv),oplt);
%         partM.visualize_vertices(uniclr('r',partM.Nv),oplt);
%     end
%     ====================== ICP TEST ======================%