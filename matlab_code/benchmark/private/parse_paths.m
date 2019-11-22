function [paths] = parse_paths(c)

outputs = list_file_names(c.path.curr_exp_dir);
assert(~isempty(outputs),"Directory %s does not contain meshes",c.path.curr_exp_dir);

% TODO - Make this system a bit smarter... 
switch c.curr_tgt_ds
    case {'faust'}
        paths = load_faust_paths(c,outputs);
    case {'amass'}
        paths = load_amass_paths(c,outputs);
    case {'dfaust'}
        error('Unimplemented');
    otherwise
        error('Unknown target dataset %s',ds_name);
end

fprintf('Found %d mesh outputs under the experiment directory\n',size(paths,1));
end
%-------------------------------------------------------------------------%
%                          
%-------------------------------------------------------------------------%
function [paths] = load_faust_paths(c, outputs)
N = length(outputs); 
paths = cell(N,5);
for i=1:N
    
    paths{i,1} = outputs{i}; % The result
    switch lower(c.curr_src_ds)
        case '3d-epn'
            % recon_tr_reg_080_001_projdata
            ids = regexp(outputs{i},'tr_reg_\d+_\d+','Match'); ids = ids{1};
            paths{i,2} = [ids , '.mat'];
            paths{i,3} = [ids(1:end-4), '.mat'];
            paths{i,4} = paths{i,3};
        case {'3d-coded','farm'}
            ids = regexp(outputs{i},'\d*','Match'); % sub,pose,sub,pose,mask
            paths{i,2} = sprintf('tr_reg_0%s%s_%03s.mat',ids{1},ids{2},ids{3}); % The part
            paths{i,3} = sprintf('tr_reg_0%s%s.mat',ids{1},ids{2}); % The Ground Truth
            paths{i,4} = sprintf('tr_reg_0%s%s.mat',ids{4},ids{5}); % The Template
        case {'litany'}
            tmp_ids = regexp(outputs{i},'\d*','Match');
            ids = cell(1,5); 
            first_id = tmp_ids{1}; ids{3} = first_id(2); ids{4} = first_id(3); 
            ids{5} = num2str(str2num(tmp_ids{2})); 
            % Use some random template subject id + pose id
            ids{1} = '1'; ids{2} = '1'; 
            fn = list_file_names(fullfile(c.path.curr_exp_dir,outputs{i})); 
            paths{i,1} = fullfile(outputs{i},fn{end}); 
            paths{i,2} = sprintf('tr_reg_0%s%s_%03s.mat',ids{3},ids{4},ids{5}); % The part
            paths{i,3} = sprintf('tr_reg_0%s%s.mat',ids{3},ids{4}); % The Ground Truth
            paths{i,4} = sprintf('tr_reg_0%s%s.mat',ids{1},ids{2}); % The Template
            
        otherwise
            
            ids = regexp(outputs{i},'\d*','Match'); % sub,pose,sub,pose,mask
            paths{i,2} = sprintf('tr_reg_0%s%s_%03s.mat',ids{3},ids{4},ids{5}); % The part
            paths{i,3} = sprintf('tr_reg_0%s%s.mat',ids{3},ids{4}); % The Ground Truth
            paths{i,4} = sprintf('tr_reg_0%s%s.mat',ids{1},ids{2}); % The Template
    end
    
    paths{i,5} = ids;
    assert(isfile(fullfile(c.path.projection_dir,paths{i,2})),"Could not find %s",paths{i,2});
    assert(isfile(fullfile(c.path.full_shape_dir,paths{i,3})),"Could not find %s",paths{i,3});
    assert(isfile(fullfile(c.path.full_shape_dir,paths{i,4})),"Could not find %s",paths{i,4}); % Not really needed
end
end

function [paths] = load_amass_paths(c, outputs)
N = length(outputs); 
paths = cell(N,5);
for i=1:N
    
    paths{i,1} = outputs{i}; % The result
    
    ids = regexp(outputs{i},'\d*','Match'); % sub,pose,sub,pose,mask
    if ~isfile(fullfile(c.path.projection_dir,sprintf('subjectID_%s_poseID_%s_projectionID_%s.npz',ids{3},ids{4},ids{5})))
        ids{1} = num2str(str2double(ids{1}) +1);% Sometimes subject ID is from 0 (Giovanni's incosistency)
        ids{3} = num2str(str2double(ids{3}) +1);
        ids{5} = num2str(str2double(ids{5}) -1);
    end
    paths{i,2} = sprintf('subjectID_%s_poseID_%s_projectionID_%s.npz',ids{3},ids{4},ids{5}); % The part
    paths{i,3} = sprintf('subjectID_%s_poseID_%s.OFF',ids{3},ids{4}); % The Ground Truth
    paths{i,4} = sprintf('subjectID_%s_poseID_%s.OFF',ids{1},ids{2}); % The Template
    
    paths{i,5} = ids;
    assert(isfile(fullfile(c.path.projection_dir,paths{i,2})),"Could not find %s",paths{i,2});
    assert(isfile(fullfile(c.path.full_shape_dir,paths{i,3})),"Could not find %s",paths{i,3});
    assert(isfile(fullfile(c.path.full_shape_dir,paths{i,4})),"Could not find %s",paths{i,4}); % Not really needed
end

end


