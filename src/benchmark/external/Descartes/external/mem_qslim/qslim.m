function [MN,C,V2W] = qslim(M,target,flags)
% QSLIM Very simple wrapper for qslim
%   The methode is based on the following paper
%       Michael Garland and Paul Heckbert, 
%       Surface Simplification Using Quadric Error Metrics, 
%       SIGGRAPH 9
% [C,W,G,V2W] = qslim(V,F,t,'ParameterName',ParameterValue)
%
% Inputs:
%   V  #V by 3 input mesh vertex positions
%   F  #F by 3 input mesh triangle indices (1-indexed)
%   t  target number of faces {0 for all collapse}
%   Optional:
%     flags
% Outputs:
%   C  #collapse list of edge collapse structs with fields
%     .a index into V of vertex collapsed to
%     .b index into V of vertex collapsed from
%     .da  1 by 3 displacement of vertex a
%     .rm list of indices in F of faces removed by this collapse
%     .mod list of indices in F of faces modified by this collapse
%   W  #W by 3 collapsed mesh vertex positions
%   G  #G by 3 collapsed mesh triangle indices (1-indexed)
%   V2W  #V list of indices into W
%
% See also: readLOG, perform_edge_collapse
%

if ~exist('flags','var'); flags = ''; end
V = M.v; F = M.f;

prefix = tempprefix();
input = [prefix '.smf'];
output = [prefix '.log'];

% Write input to file
write_smf(input,V,F);

% prepare command string
command = sprintf('"%s" -t %d -M log -q %s %s >%s', ...
    which('QSlim.exe'),target,flags,input,output);
try
    [status,result] = system(command);
    if status ~= 0
        error(result);
    end
    
    [VV,FF,C] = readLOG(output);
    % should match input
    assert(size(VV,1)==size(V,1));
    assert(size(FF,1)==size(F,1));
    assert(isequal(F,FF));
    
    delete(input);
    delete(output);
catch ME
    fprintf('%s\n',command);
    rethrow(ME);
end

[W,G,V2W] = perform_edge_collapse(V,F,C,target);
MN = Mesh(W,G,M.name,M.path);

end
