function [fn,fp] = list_file_names(dir_name)
    d=dir(dir_name);
    fn={d(~ismember({d.name},{'.','..'})).name};
    % Remove ASV & Desktop.ini files 
    fn = fn(~endsWith(fn,'.asv')); 
    fn = fn(~strcmp(fn,'desktop.ini')); 
    
    assert(~isempty(fn),sprintf('Empty directory : \n%s',dir_name)); 
    fn = natsortfiles(fn).'; 
    fp = fullfile(dir_name,fn); 
end