function [paths,stats] = filter_subset(c,stats,paths)

if strcmp(c.sort_meth,'none'); return; end
banner('Path Filteration');
N = size(paths,1);

% First Step - Filter out paths
goners = false(N,1);
if isfield(c,'pick_n_at_random') && c.pick_n_at_random > 0
    goners(:) = true;
    goners(randsample(size(paths,1),c.pick_n_at_random)) = false;
else
    for i=1:N
        if c.no_self_match && strcmp(paths{i,3},paths{i,4}) % When the template is exactly the ground truth
            goners(i) = true;
            continue;
        end
        if isfield(c,'look_at_sub') && ~isempty(c.look_at_sub)
            ids = paths{i,5}; % Presuming same subject id for both shapes
            if ~any(ismember(ids(1),c.look_at_sub))
                goners(i) = true;
                continue;
            end
        end
        if isfield(c,'look_at_template_pose') && ~isempty(c.look_at_template_pose)
            ids = paths{i,5};
            if ~any(ismember(ids(2),c.look_at_template_pose))
                goners(i) = true;
                continue;
            end
        end
        if isfield(c,'look_at_projection_pose') && ~isempty(c.look_at_projection_pose)
            ids = paths{i,5};
            if ~any(ismember(ids(4),c.look_at_projection_pose))
                goners(i) = true;
                continue;
            end
        end
        if isfield(c,'look_at_projection_id') && ~isempty(c.look_at_projection_id)
            ids = paths{i,5};
            if ~any(ismember(ids(5),c.look_at_projection_id))
                goners(i) = true;
                continue;
            end
        end
    end
end

% Update the data structures
paths(goners,:) = [];
assert(~isempty(paths),'Filtered out all files - None are left');
stats = trunc_struct(stats,N,~goners);
N = size(paths,1);
fprintf('Remained with %d subjects after filtering\n',N);


% Second Step - Sort the remaining paths by some criterion
switch c.sort_meth
    case 'seq' % Do nothing
        idx = 1:N;
        fprintf('Sorted sequentially\n');
    case 'highestL2'
        [~,idx] = sort(stats.me_err,'descend');
        fprintf('Sorted by Euclidean V2V error - Highest -> Lowest\n');
    case 'lowestL2'
        [~,idx] = sort(stats.me_err,'ascend');
        fprintf('Sorted by Euclidean V2V error - Lowest -> Highest\n');
    case 'rand'
        idx = randperm(N);
        fprintf('Randomly shuffled the path set\n');
    case 'lowest_chamfergt2res'
        [~,idx] = sort(stats.chamfer_gt2res,'ascend');
        fprintf('Sorted by Chamfer GT->Res - Lowest -> Highest\n');
    case 'lowest_chamferres2gt'
        [~,idx] = sort(stats.chamfer_res2gt,'ascend');
        fprintf('Sorted by Chamfer GT->Res - Lowest -> Highest\n');
end
paths = paths(idx,:);
stats = trunc_struct(stats,N,idx);

if c.export_subset
    export_subset(c,paths,stats)
end

report_moments('Euclidean V2V',stats.me_err,'cm');
report_moments('Volume',stats.vol_err,'%');
report_moments('Chamfer GT->Result',stats.chamfer_gt2res,'cm');
report_moments('Chamfer Result->GT',stats.chamfer_res2gt,'cm');
report_moments('Correspondence Direct Hits',stats.correspondence_10_hitrate,'%');

end


