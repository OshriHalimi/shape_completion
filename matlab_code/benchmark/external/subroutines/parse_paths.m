function [c,paths] = parse_paths(c)

outputs = list_file_names(c.curr_exp_dir);
assert(~isempty(outputs),"Directory %s does not contain meshes",c.curr_exp_dir);
% Locate dataset name

lastpos = find(c.exp_name == '2', 1, 'last');
c.ds_name = lower(c.exp_name(lastpos+1:end)); 

% if ~isfield(c,'ds_name')
%     [ds_name] = split(outputs{1},'_'); c.ds_name = ds_name{1};
% end
switch c.ds_name
    case {'faust'}
        c.full_shape_dir = fullfile(c.data_dir,'faust_projections','dataset');
        c.projection_dir = c.full_shape_dir; 
        %         c.triv_dir = fullfile(c.data_dir,'faust_projections','range_data','res=100x180');
        paths = load_faust_paths(c,outputs);
    case {'amass'}
        c.full_shape_dir = fullfile(c.data_dir,'amass','test','original');
        c.projection_dir = fullfile(c.data_dir,'amass','test','projection');
        paths = load_amass_paths(c,outputs);
    otherwise
        error('Unknown dataset %s',ds_name);
end

fprintf('Found %d mesh sets\n',size(paths,1));
end


function [paths] = load_faust_paths(c, outputs)
% {'faust_completion_subjectIDfull_8_poseIDfull_0_subjectIDpart_8_poseIDpart_0_mask_1.mat'}
% faust_completion_subjectIDfull_8_poseIDfull_0_subjectIDpart_8_poseIDpart_0_mask_1
N = length(outputs); paths = cell(N,5);
for i=1:N
    ids = regexp(outputs{i},'\d*','Match'); % sub,pose,sub,pose,mask
    paths{i,1} = outputs{i}; % The result
    paths{i,2} = sprintf('tr_reg_0%s%s_%03s.mat',ids{3},ids{4},ids{5}); % The part
    paths{i,3} = sprintf('tr_reg_0%s%s.mat',ids{3},ids{4}); % The Ground Truth
    paths{i,4} = sprintf('tr_reg_0%s%s.mat',ids{1},ids{2}); % The Template
    paths{i,5} = ids; 
    assert(isfile(fullfile(c.projection_dir,paths{i,2})),"Could not find %s",paths{i,2});
    assert(isfile(fullfile(c.full_shape_dir,paths{i,3})),"Could not find %s",paths{i,3});
    assert(isfile(fullfile(c.full_shape_dir,paths{i,4})),"Could not find %s",paths{i,4}); % Not really needed
end

end

function [paths] = load_amass_paths(c, outputs)
% {'faust_completion_subjectIDfull_8_poseIDfull_0_subjectIDpart_8_poseIDpart_0_mask_1.mat'}
% faust_completion_subjectIDfull_8_poseIDfull_0_subjectIDpart_8_poseIDpart_0_mask_1
N = length(outputs); paths = cell(N,5);
for i=1:N
    ids = regexp(outputs{i},'\d*','Match'); % sub,pose,sub,pose,mask
    if ~isfile(fullfile(c.projection_dir,sprintf('subjectID_%s_poseID_%s_projectionID_%s.npz',ids{3},ids{4},ids{5})))
        ids{1} = num2str(str2double(ids{1}) +1);% Sometimes subject ID is from 0 (Giovanni's incosistency) 
        ids{3} = num2str(str2double(ids{3}) +1);
        ids{5} = num2str(str2double(ids{5}) -1); 
    end
    paths{i,1} = outputs{i}; % The result
    paths{i,2} = sprintf('subjectID_%s_poseID_%s_projectionID_%s.npz',ids{3},ids{4},ids{5}); % The part
    paths{i,3} = sprintf('subjectID_%s_poseID_%s.OFF',ids{3},ids{4}); % The Ground Truth
    paths{i,4} = sprintf('subjectID_%s_poseID_%s.OFF',ids{1},ids{2}); % The Template
    paths{i,5} = ids; 
    assert(isfile(fullfile(c.projection_dir,paths{i,2})),"Could not find %s",paths{i,2});
    assert(isfile(fullfile(c.full_shape_dir,paths{i,3})),"Could not find %s",paths{i,3});
    assert(isfile(fullfile(c.full_shape_dir,paths{i,4})),"Could not find %s",paths{i,4}); % Not really needed
end

end


