function write_smf(filename, vertex, face, renormalize)

% write_smf - write a mesh to a SMF file
%
%   write_smf(filename, vertex, face);
%
%   vertex must be of size [n,3]
%   face must be of size [p,3]
%
%   Copyright (c) 2003 Gabriel Peyr�


if nargin<4
    renormalize = 0;
end

if size(vertex,2)~=3
    vertex=vertex';
end
if size(vertex,2)~=3
    error('vertex does not have the correct format.');
end

if renormalize==1
    m = mean(vertex);
    s = std(vertex);
    for i=1:3
        vertex(:,i) = (vertex(:,i)-m(i))/s(i);
    end
end

if size(face,2)~=3
    face=face';
end
if size(face,2)~=3
    error('face does not have the correct format.');
end

fid = fopen(filename,'wt');
if( fid==-1 )
    error('Can''t open the file.');
end

% header
fprintf(fid, '#$SMF 1.0\n');
fprintf(fid, '#$vertices %d\n#$faces %d\n', size(vertex,1), size(face,1));

% write the points & faces
fprintf(fid, 'v %f %f %f\n', vertex');
fprintf(fid, 'f %d %d %d\n', face');

fclose(fid);