function [paths,stats] = sort_paths(c,stats,paths)

N = size(paths,1); 
goners = false(N,1);
if isfield(c,'pick_n_at_random') && c.pick_n_at_random > 0 
    goners(:) = true; 
    goners(randsample(size(paths,1),c.pick_n_at_random)) = false; 
else
    for i=1:N
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

        if c.no_self_match && strcmp(paths{i,3},paths{i,4}) % When the template is exactly the ground truth
            goners(i) = true;
        end
    end
end
paths(goners,:) = [];
stats.(c.exp_name).ME_err(goners) = []; 
stats.(c.exp_name).chamfer_gt2res(goners) = []; 
stats.(c.exp_name).chamfer_res2gt(goners) = []; 
assert(~isempty(paths),'Filtered out all files - None are left');
fprintf('Remained with %d subjects after filtering\n',size(paths,1));

switch c.sort_meth
    case 'seq' % Do nothing
    case 'highestL2'
        [~,idx] = sort(stats.(c.exp_name).ME_err,'descend');
        paths = paths(idx,:);
    case 'lowestL2'
        [~,idx] = sort(stats.(c.exp_name).ME_err,'ascend');
        paths = paths(idx,:);
    case 'rand'
        paths = paths(randperm(size(paths,1)),:);
        fprintf('Shuffled the shape sets\n');
end
end