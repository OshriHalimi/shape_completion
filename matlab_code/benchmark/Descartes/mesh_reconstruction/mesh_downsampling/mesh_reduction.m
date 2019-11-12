function [MR] = mesh_reduction(M,meth,o)
    reducepatch
    if i==N_obj || type{i,2}~=type{i+1,2} %object is the first in the class
        [C,W,G,V2W] = qslim(v,triv,20000,'QSlimFlags','-O 0');
        [W2V] = calcW2V(W,v,V2W);
    end
        r = 6000/n; %sampling reduction factor
    if r<1
        nfv = reducepatch(triv,v,r);
        surface = nfv;
    
    nfv.faces = G;
    nfv.vertices = v(W2V,:);
end

function [W2V] = calcW2V(W,V,V2W)
    W2V = -ones(size(W,1),1);
    mindist = inf(size(W,1),1);
    for i=1:size(V,1)
        dist = norm(V(i,:)-W(V2W(i),:));
        if dist<mindist(V2W(i))
            mindist(V2W(i)) = dist;
            W2V(V2W(i)) = i;
        end
    end
end

