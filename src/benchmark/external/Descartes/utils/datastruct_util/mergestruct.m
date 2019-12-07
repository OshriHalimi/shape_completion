function into = mergestruct(into, from,force)
%MERGESTRUCT merge all the fields of scalar structure from into scalar structure into
if(~exist('force','var'))
    force =0;
end
validateattributes(from, {'struct'}, {'scalar'});
validateattributes(into, {'struct'}, {'scalar'});
fns = fieldnames(from);
for fn = fns.'
    if  ~isfield(into, fn{1}) && force == 0
        error('Detected unknown field "%s" from source struct', fn{1});
    end
    if isstruct(from.(fn{1})) && (force == 2 || isfield(into, fn{1}))
        %nested structure where the field already exist, merge again
        into.(fn{1}) = mergestruct(into.(fn{1}), from.(fn{1}),force);
    else
        %non structure field, or nested structure field that does not already exist, simply copy
        into.(fn{1}) = from.(fn{1});
    end
end
end