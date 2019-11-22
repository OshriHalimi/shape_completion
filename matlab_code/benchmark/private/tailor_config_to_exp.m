function [c] = tailor_config_to_exp(c,exp_name)

c.curr_exp = exp_name; 

c.path.curr_exp_dir = fullfile(c.path.exps_dir,c.curr_exp);

% Parse the source and target dataset names
pos = find(c.curr_exp == '2', 1, 'last'); 
assert(~isempty(pos),"Invalid experiment name %s",c.curr_exp); 
c.curr_tgt_ds = lower(c.curr_exp(pos+1:end)); 

pos2 = find(c.curr_exp == '_', 1, 'first'); 
assert(~isempty(pos2),"Invalid experiment name %s",c.curr_exp); 
c.curr_src_ds = lower(c.curr_exp(pos2+1:pos-1)); 


switch c.curr_tgt_ds
    case {'faust'}
        c.path.full_shape_dir = fullfile(c.path.data_dir,'faust_projections','dataset');
        c.path.projection_dir = c.path.full_shape_dir;
    case {'amass'}
        c.path.full_shape_dir = fullfile(c.path.data_dir,'amass','test','original');
        c.path.projection_dir = fullfile(c.path.data_dir,'amass','test','projection');
    case {'dfaust'}
        error('Unimplemented');
    otherwise
        error('Unknown target dataset %s',ds_name);
end


end