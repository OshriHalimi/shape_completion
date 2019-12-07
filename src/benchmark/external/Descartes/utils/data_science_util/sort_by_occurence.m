function [Vs,cnts] = sort_by_occurence(V,dir)
% The following code first calculates how often each element occurs and 
% then uses runLengthDecode to expand the unique elements.
if ~exist('dir','var'); dir = 'descend'; end
Vu = unique(V);
cnts = histc(V,Vu);
[cnts, idx] = sort(cnts,dir);
Vs = repelem(Vu(idx), cnts);
end
