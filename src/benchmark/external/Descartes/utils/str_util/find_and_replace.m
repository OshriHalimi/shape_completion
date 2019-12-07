function find_and_replace(file, expression, replacement,mock)
% find_and_replace(file, to_find, to_replace)
%
% USAGE: find_and_replace(get_repo(),'uniform_clr\((.*?)\)','uniclr\($1\)',1)
%
% Replaces a string (expression) in a given file (file) with a new string
% (replacement).
%
% See regexprep for info on valid 'expression' and 'replace' inputs.
%
% Examples:
%
% Replace all instances of the string 'abs', including 'cabs', with 'mag':
%
% find_and_replace('my_file.m', 'abs', 'mag');
%
% Replace all the exact variable names 'my_var' with 'my_other_var'. This
% *won't* match 'my_var_2' or 'this_is_my_var'.
%
% find_and_replace('my_file.m', '\<my_var\>', 'my_other_var');
%
% Replace all calls to sqrt(...) with my_sqrt(...), keeping the function's
% argument (this is using regular expressions).
%
% find_and_replace(tc.file_name, 'sqrt\((.*?)\)', 'my_sqrt\($1\)');
%
% Note that the above does *not* work for nested parentheses (regular
% expressions can't match nested parentheses to an arbitrary depth).
%
% This function can also handle multiple files, where the first input is
% either a cell array of file names (e.g., {'file1.txt', 'file2.m'}) or a
% struct with a 'name' field that contains the file name, such as is output
% from files = dir('*.m');
%
% For example, to replace in all .m files in the current directory:
%
% find_and_replace(dir('*.m'), 'string_1', 'string_2');
%
% See doc regexprep for more examples.
%
% Tucker McClure
% Copyright 2013, The MathWorks, Inc.
if ~exist('mock','var'); mock = 0; end

% If user passed in a single file name, wrap it as a cell.
if ischar(file)
    file = {file};
elseif isstruct(file) && any(strcmp(fieldnames(file), 'name'))
    file = {file(:).name};
elseif ~iscell(file)
    error('find_and_replace:invalid_inputs', ...
        'Unknown input type for ''file''.');
end

% For all files in the cell array...
if mock; matches = {}; n_tgtf = 0; end
for k = 1:length(file)
    
    % Make sure the file exists.
    if ~exist(file{k}, 'file')
        error('find_and_replace:no_file', ...
            ['File doesn''t exist. To replace strings in text, ' ...
            'use regexprep.']);
    end
    
    
    if mock
        lines = {}; 
        fid = fopen(file{k}); found = 0; ln = 1; 
        tline = fgetl(fid);   
        while ischar(tline)
          c = regexp({tline},expression);
          if ~isempty(c{:})
              found = found+1; 
              lines{found,1} = ln;
              lines{found,2} = tline;
          end
          tline = fgetl(fid);ln = ln+1;
        end
        if ~isempty(lines)
            n_tgtf = n_tgtf+1; 
            matches{n_tgtf,1} = file{k};
            matches{n_tgtf,2} = lines;
        end
        fclose(fid);
    else
        % Read in the file as binary and convert to chars.
        fid = fopen(file{k});
        lines = fread(fid, inf, '*char')';
        fclose(fid);
        
        % Find and replace.
        lines = regexprep(lines, expression, replacement);
        
        % Write out the new file.
        fid = fopen(file{k}, 'w');
        fwrite(fid, lines);
        fclose(fid);
    end
end

end
