function [menu] = test_database_status(exclude)

menu = test_mesh(); 
if ~exist('exclude','var'); exclude = {};end
fns = fieldnames(menu);
for i=1:numel(fns)
    % Check for lib skip: 
    if any(strcmp(fns{i},exclude))
        cprintf('*Blue',usprintf('---- SKIPPED: %s Library ----\n',fns{i}));
        continue;
    end
    % Dereference lib: 
    lib = menu.(fns{i});
    cprintf('*Blue',usprintf('---- %s Library ----\n',fns{i}));
    for j=1:numel(lib)
        test_mesh(lib{j});
    end
end
end
