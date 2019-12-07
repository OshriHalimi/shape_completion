function fileList = findfiles(searchPath,filePattern,patternMode,fileList)
% FUNCTION files = RECURSIVEFINDFILE(searchPath,filePattern)
% A case insensitive search to find files with in the folder _searchPath_ 
% (if specified) whos file name matches the pattern _filePattern_ and add 
% the results to the list of files in the cell array _fileList_. 
%
% The search can be either a wild-card or regular expression search and is
% case insenstive
%
%
% INPUTS (all are optional
%    searchPath  - Default ask user
%                - directory or string to search can be either a absolute 
%                  or relative path.
%                
%    filePattern - Default: *.* (All files)
%                - List of file wildcard file patterns to search
%                - Wild-card examples
%                     filePattern = '*.xls' % find Excel files
%                     filePattern = {'*.m' *.mat' '*.fig'}; % MATLAB files
%
%                - Regular expression examples (See regexpi help for more)
%                     filePattern = '^[af].*\.xls' % Excel files beginning
%                                                  % with either A,a,F or f
%
%    patternMode - Default: 'Wildcard'
%                - 'Wildcard' for wild-card searches or 
%                - 'Regexp' for regular expression searches
%    fileList    - list of files (nx1 cell). Default is an empty cell.
   
%
%
% author: Azim Jinha (2011)
%% Test inputs
%*** searchPath ***
if ~exist('searchPath','var') || isempty(searchPath) || ~exist(searchPath,'dir')
    searchPath = uigetdir('Select Path to search');
end
% *** filePattern ***
if ~exist('filePattern','var') || isempty(filePattern), filePattern={'*.*'}; end
if ~iscell(filePattern)
    % if only one file pattern is entered make sure it 
    % is still a cell-string.
    filePattern = {filePattern};
end
% *** patternMode ***
if ~exist('patternMode','var')||isempty(patternMode),patternMode = 'wildcard'; end
switch lower(patternMode)
case 'wildcard'
    % convert wild-card file patterns to regular expressions
    fileRegExp=cell(length(filePattern(:)));
    for i=1:length(filePattern(:))
        fileRegExp{i}=regexptranslate(patternMode,filePattern{i});
    end
otherwise
    % assume that the file pattern(s) are regular expressions
    fileRegExp = filePattern;
end
% *** fileList ***
% test input argument file list
if ~exist('fileList','var'),fileList = {}; end % does it exist
% is fileList a nx1 cell array
if size(fileList,2)>1, fileList = fileList'; end 
if ~isempty(fileList) && min(size(fileList))>1, error('input fileList should be a nx1 cell array'); end
%% Perform file search
% Get the parent directory contents
dirContents = dir(searchPath);
for i=1:length(dirContents)
    if ~strncmpi(dirContents(i).name,'.',1)
        newPath = fullfile(searchPath,dirContents(i).name);
        if dirContents(i).isdir
            fileList = findfiles(newPath,filePattern,patternMode,fileList);
        else
            foundFile=false;
            for jj=1:length(fileRegExp)
                foundFile = ~isempty(regexpi(dirContents(i).name, ...
                                             fileRegExp{jj}));
                if foundFile, break; end
            end
            if foundFile
                fileList{end+1,1} = newPath; %#ok<AGROW>
            end
        end
    end
end