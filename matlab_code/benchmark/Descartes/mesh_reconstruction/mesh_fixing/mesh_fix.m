function [MN] = mesh_fix(M,verbose,flags)

if ~exist('flags','var'); flags = '';end
if ~exist('verbose','var'); verbose = 0;end


prefix = tempname;
off_filename = [prefix '.off'];
off_filename_fixed = [prefix '_fixed.off'];
M.export_as(off_filename);

command = ['"' which('MeshFix.exe') '"' ' ' flags ' ' off_filename];
[status, result] = system(command);
if status ~= 0
    fprintf(command);
    error(result);
elseif verbose
    disp(command);
    disp(result);
end

MN = read_mesh(off_filename_fixed);
MN.name = M.name; MN.file_name = M.file_name; MN.path = M.path;
MN.plt.title = M.name; 

delete(off_filename);
delete(off_filename_fixed);
end
