function [MN,ind1,ind2]=trim_close_vertices(M,mode,tol)

% TODO - Implement simply 'distance' and neighbor dec_pts
switch lower(mode)
    case 'neighbor_distance' 
        if ~exist('tol','var') || isempty(tol); tol = 1e-8; end
        [D,E] = M.edge_distances();
        too_short = find(D < tol);
        % Turn the 2nd id into the first
        query = M.v;
        for i=1:length(too_short) % This is in a loop due to the sorted order of the edges
            goner_vs = E(too_short(i),2);
            query(goner_vs,:) = M.v(E(too_short(i),1),:);
        end
    case 'dec_pts'
        if ~exist('tol','var') || isempty(tol)
            E = M.edges();
            D = vecnorm(M.v(E(:,1),:)-M.v(E(:,2),:),2,2);
            tol=6-num_order(mean(D));
        end
        query = round(M.v,tol);
    otherwise
        error('Unimplemented mode %s',mode);
end

% Merge nodes
[v2,ind1,ind2]=unique(query,'rows');
%Fix indices in F
f2=ind2(M.f); 
% In some cases, this can create degenerate faces with two vertices that
% are the same 
f20 = sort(f2,2);
f2 = f2(sum([ones(size(f2,1),1),diff(f20,[],2)~=0],2) >= 3,:);

% Destroy redundancies: 
[v2,f2] = trim_mesh_redundancies(v2,f2);

MN = Mesh(v2,f2,M.name,M.path);
end

