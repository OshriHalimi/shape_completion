function mesh_montage(ms,opt,plt_override)

% Defaults
o.tight = [];
o.dset_name = [];
o.max_single_disp = 16;
o.max_multi_disp = 8;
o.perm = 0;
o.disp_func = [];
o.C = []; %Might be a cell array , or a matrix
o.names = [];
o.use_names = 1;

% Defaults override
if exist('opt','var')
    o = mergestruct(o,opt); % Add in override
    if ~isempty(o.names)
        o.use_names = 1;
    end
end
if ~exist('plt_override','var')
    plt_override = struct();
end

if o.perm
    % TODO - Insert support for random names as well
    mesh_idx = randperm(numel(ms));
    ms = ms(mesh_idx);
end

% Open up the figure
fullfig; axis('off')
% setappdata(gcf, 'SubplotDefaultAxesLocation', [0, 0, 1, 1]); % Enlarges
% each figure, but leaves no room for the suptitle sometimes.
if ~isempty(o.dset_name);suptitle(o.dset_name);end

n_mesh = numel(ms);
n_clrs = max(size(o.C,2),1);
% Display all params:
if n_clrs == 1 || n_mesh == 1
    montage_single(ms,o,plt_override);
else
    montage_mult(ms,o,plt_override);
end

function montage_single(ms,o,plt_override)

n_mesh = numel(ms);
n_clrs = size(o.C,2);

n_elems = min(max(n_mesh,n_clrs),o.max_single_disp);
n_rows = floor(sqrt(n_elems));
n_cols = ceil(n_elems/n_rows);

if iscell(ms)
    M = ms{1};
else
    M = ms;
end
M.plt.new_fig = 0;
C = [];

for i=1:n_elems
    if isempty(o.tight)
        subplot(n_rows,n_cols,i);
    else
        subplot_tight(n_rows,n_cols,i,o.tight);
    end
    if n_mesh == 1 && n_clrs > 0 % Displaying Color
        C = o.C(:,i);
    else %Displaying Mesh
        M = ms{i};M.plt.new_fig = 0;
    end
    if o.use_names
        if ~isempty(o.names)
            M.plt.title = o.names{i};
        end
        % Default - Use Mesh Names
    else
        M.plt.title = '';
    end
    if ~isempty(o.disp_func)
        o.disp_func(M,C,plt_override);
    else
        M.visualize(C,plt_override);
    end
    
end
end
end

function montage_mult(ms,o,plt_override)

n_mesh = numel(ms);
if iscell(o.C)
    n_clrs = max(size(o.C{1},2),1);
else
    n_clrs = max(size(o.c,2),1);
end
n_rows = min(n_mesh,o.max_multi_disp);
n_cols = min(n_clrs,o.max_multi_disp);

for i=1:n_rows
    M = ms{i}; M.plt.new_fig = 0;
    if iscell(o.C)
        CMat = o.C{i};
    else
        CMat = o.C;
    end
    for j=1:n_cols
        C = CMat(:,j);
        linind = sub2ind([n_cols,n_rows],j,i);
        if isempty(o.tight)
            subplot(n_rows,n_cols,linind);
        else
            subplot_tight(n_rows,n_cols,linind,o.tight);
        end
        if o.use_names
            if ~isempty(o.names)
                M.plt.title = o.names{linind};
            end
            % Default - Use Mesh Names
        else
            M.plt.title = '';
        end
        if ~isempty(o.disp_func)
            o.disp_func(M,C,plt_override);
        else
            M.visualize(C,plt_override);
        end
    end
end
end
