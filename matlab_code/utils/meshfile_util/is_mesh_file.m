function [ismf] = is_mesh_file(name)

[~,~,ext] = fileparts(name); 
ext = lower(ext);
if isempty(ext)
    ismf = false;
else
    ext = ext(2:end); % Remove dot
    ismf = any(strcmp(ext,supported_ext(1)));
end
end

