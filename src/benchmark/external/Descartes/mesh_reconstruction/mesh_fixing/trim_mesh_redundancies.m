function [v,f] = trim_mesh_redundancies(v,f)
Nv = 0; Nf = 0; iters = 0;
while(Nv ~= size(v,1) || Nf ~= size(f,1))
    iters = iters+1;
    Nv = size(v,1); Nf = size(f,1);
    f = trim_face_redundancies(f);
    [v,f] = trim_vertex_redundancies(v,f);
end
assert(iters <=2); % TODO - While loop might be redundant
end
%-------------------------------------------------------------------------%
%                           Global Config
%-------------------------------------------------------------------------%
function f = trim_face_redundancies(f)

sf = sort(f, 2);
[~,I] = unique(sf,'rows','stable');
dup_ind = setdiff(1:size(f, 1), I);
if numel(dup_ind)
    fprintf('DEBUG: Found %d redundant faces\n',numel(dup_ind)); 
    f(dup_ind,:) = [];
end
end

function [v2,f2] = trim_vertex_redundancies(v,f)
% Delete duplicate vetices
[tempVertices, ~, tempFaceVertexIdx] = unique(v, 'rows');
tempFaces = tempFaceVertexIdx(f);
% Delete unindexed/unreferenced vertices
usedVertexIdx = ismember(1:length(tempVertices),unique(tempFaces(:)));
newVertexIdx = cumsum(usedVertexIdx);
faceVertexIdx = 1:length(tempVertices);
faceVertexIdx(usedVertexIdx) = newVertexIdx(usedVertexIdx);
faceVertexIdx(~usedVertexIdx) = nan;
f2 = faceVertexIdx(tempFaces);
v2 = tempVertices(usedVertexIdx,:);
end

% Nf = size(f, 1);
% goners = false(Nf, 1);
% for fi = 1:Nf
%     if goners(fi); continue; end
%
%     face = f(fi, :);
%     inds = find(sum(ismember(f, face), 2) == 3);
%     inds(inds <= fi) = []; % Look only at indices higher than fi
%     if ~isempty(inds); goners(inds) = true; end
% end
%
% f2 = f(~goners, :);
% sum(goners)
