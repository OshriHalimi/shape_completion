function N = numel_struct(s)
%MERGESTRUCT merge all the fields of scalar structure from into scalar structure into
N = 0; 
validateattributes(s, {'struct'}, {'scalar'});
fns = fieldnames(s);
for fn = fns.'
    if isstruct(s.(fn{1})) 
        N = N + mergestruct(s.(fn{1}));
    else
        %non structure field, or nested structure field that does not already exist, simply copy
        N = N + numel(s.(fn{1})); 
    end
end
end