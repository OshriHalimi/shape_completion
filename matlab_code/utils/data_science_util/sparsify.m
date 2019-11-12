function [sdata,ids] = sparsify(data,pick)
% data = Original Data 
% pick = Either the ids (cell vector) or a scalar between 0 and 1 detailing the
% fraction of the data that is to be kept or a number greater 1 detailing
% the maximal number of points to keep.
% If pick is not chosen, we keep 0.05% of the data

if ~exist('pick','var') || isempty(pick)
    pick = 0.05;
end

n = size(data,1);
if iscell(pick)
    ids = pick{1};
elseif ~isscalar(pick)
    ids = pick;
else % Presume isscalar
    if pick > 0 && pick <= 1
        k = floor(pick * n);
        ids = randperm(n,k);
    elseif pick > 1
        ids = randperm(n,min(pick,n));
    else
        % Don't sparsify
        ids = 1:n; 
    end
end
sdata = data(ids,:);
end
