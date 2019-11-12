function [renames] = assimilate_test_database(exclude,mock)
if ~exist('exclude','var'); exclude = {};end
if ~exist('mock','var'); mock = 1;end

renames = align_test_dataset(exclude);
% Do the actual renaming (Fixed points were removed)
if ~mock
    for i=1:size(renames,1)
        if ~isfile(renames{i,1})
            warning('No such file: %s - Was it moved?',renames{i,1});
            continue;
        end
        movefile(renames{i,1},renames{i,2},'f');
    end
end
end

function renames = align_test_dataset(exclude)
menu = test_mesh();fns = fieldnames(menu);
N = numel_struct(menu);
renames = cell(N,2);
idx = 1;
for i=1:numel(fns)
    if any(strcmp(fns{i},exclude))
        cprintf('*Blue',usprintf('---- SKIPPED: %s Library ----\n',fns{i}));
        continue;
    end
    lib = menu.(fns{i});cprintf('*Blue',usprintf('---- %s Library ----\n',fns{i}));
    for j=1:numel(lib)
        [prior,tgt]=compute_rename_target(lib{j});
        if ~strcmp(prior,tgt) % No need to rename perfect matches
            renames{idx,1}=prior;
            renames{idx,2}=tgt;
            idx = idx +1;
        end
    end
end

% Run some checkups:
content_cnt = sum(cellfun(@isempty,renames),2);
types = unique(content_cnt);
assert(numel(types)==2 && sum(types)==2);
empty_cell_id = find(content_cnt, 1, 'first')-1;
% Truncate empty cells
renames = renames(1:empty_cell_id,:);
% Make sure we did not run over any cells:
assert(numel(unique(renames(:,2)))==numel(renames(:,2)));
end

function [prior,tgt] = compute_rename_target(mesh_file_name)

[M,~,~,stat] = test_mesh(mesh_file_name);
% Check the file name originates from test dataset:
s = regexp(mesh_file_name,'(_mini[DbSPs]*\.|_\d+k[DbSPs]*\.)', 'once');
if isempty(s)
    assert(false,sprintf('Invalid name at: %s',mesh_file_name));
end
% Remove extension & flags:
[~,mesh_name_flags,~] = fileparts(mesh_file_name);
[mesh_name,~]=regexp(mesh_name_flags,'(.*)\_.*$','tokens','match'); %Equivalent to split?
assert(~isempty(mesh_name)); mesh_name = mesh_name{1};

% Compute Ender:
Nf = round(M.Nf,-3)/1000;
if Nf == 0
    ender = sprintf('mini');
else
    ender = sprintf('%dk',Nf);
end
if stat(1)
    if stat(1) > 0.05
        ender = [ender, 'SS'];
    else
        ender = [ender, 'S'];
    end
end
if stat(3)~=1
    if stat(3) > 10
        ender = [ender, 'DD'];
    else
        ender = [ender, 'D'];
    end
end
if stat(2)
    if stat(2) > 0.05
        ender = [ender, 'bb'];
    else
        ender = [ender, 'b'];
    end
end
[~,~,ext] = fileparts(mesh_file_name); assert(~isempty(ext));
tgtname = sprintf('%s_%s%s',mesh_name{1},ender,ext);
prior = which(mesh_file_name);
tgt = fullfile(fileparts(prior),tgtname);
% cprintf('*Text','%s->%s\n',prior,tgt);
cprintf('*Text','%s->%s\n',mesh_file_name,tgtname);
end


