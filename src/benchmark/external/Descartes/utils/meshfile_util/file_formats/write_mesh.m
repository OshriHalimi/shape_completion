function write_mesh(M,file)

ext = file(end-2:end);
switch lower(ext)
    case 'mat'
        v = M.v; f = M.f; 
        save(file,'v','f'); 
    case 'off'
        write_off(file, M.v, M.f);
    case 'obj'
        write_obj(file, M.v, M.f);
    case 'ply' 
        write_ply(file, M.v, M.f);
    case 'smf'
        write_smf(file, M.v, M.f);
    case 'wrl'
        write_wrl(file, M.v, M.f);
    case 'stl'
        write_stl(file, M.f, M.v);
    otherwise
        error('Unknown extension.');    
end