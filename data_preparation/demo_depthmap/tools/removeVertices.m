function N = removeVertices(N,v,apply_to_gt)
    tri2keep = sum(v(N.TRIV),2)==0;
    N.TRIV = N.TRIV(tri2keep,:);
    N.VERT = N.VERT(~v,:);
    reindex(~v)=1:sum(~v);
    N.TRIV = reindex(N.TRIV); 
    if (nargin==2) || (nargin==3 && apply_to_gt)
        N.gt = N.gt(~v);
    end
    N.m = size(N.TRIV,1);
    N.n = size(N.VERT,1);
end
