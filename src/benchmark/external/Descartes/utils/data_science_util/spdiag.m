function [D] = spdiag(d,k)
if ~exist('k','var'); k = 0; end 
n = length(d) +abs(k);
d = d(:);
if k>0
    d = [zeros(k,1);d];
elseif k<0
    d = [d;zeros(abs(k),1)]; 
end
D = spdiags(d,k,n,n);
end