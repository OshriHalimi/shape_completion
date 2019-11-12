function [f] = triangulate_faces(qf)

if iscell(qf)
    f = general_tri(qf);
else
    switch size(qf,2)
        case 3
            f = qf;
        case 4
            f = [qf(:,[1,2,4]);qf(:,[2,3,4])];
        case 5
            f = [qf(:,[1,2,3]);qf(:,[1,3,4]);qf(:,[1,4,5])];
            
        otherwise
            error('Unimplemented triangulation on %d edges',size(qf,2));
    end
end

function [f,inds] = general_tri(qf)
%   Also returns original face index of each new triangular face. INDS has
%   the same number of rows as TRI, and has values between 1 and the
%   number of rows of the original FACES array.
nf =length(qf);
% compute total number of triangles
ni = zeros(nf, 1);
for i = 1:nf
    % as many triangles as the number of vertices minus 1
    ni(i) = length(qf{i}) - 2;
end
nt = sum(ni);

% allocate memory for triangle array
f = zeros(nt, 3);
inds = zeros(nt, 1);

% convert faces to triangles
t = 1;
for i = 1:nf
    face = qf{i};
    nv = length(face);
    v0 = face(1);
    for j = 3:nv
        f(t, :) = [v0 face(j-1) face(j)];
        inds(t) = i;
        t = t + 1;
    end
end