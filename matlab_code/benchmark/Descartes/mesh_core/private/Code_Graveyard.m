%-------------------------------------------------------------------------%
%                                Graveyard
%-------------------------------------------------------------------------%
% Old Face Set:
% face_set = [v(f(:,1),:),v(f(:,2),:),v(f(:,3),:)];
% Old Adjacency Matrix
%         function test(M)
%             assert('Unimplemented');
%             V = load('../test_data/feline.mat');
%             A = false(size(v,1));
%             for i = 1:size(f,1)
%                 A(f(i,1),f(i,2)) = true;
%                 A(f(i,2),f(i,3)) = true;
%                 A(f(i,3),f(i,1)) = true;
%             end
%             A = full(max(A,A.'));
%         end
%                 tic;
%                 M = M.add_face_ring();
%                 M.va = zeros(M.Nv,1);
%                 for i=1:M.Nv
%                     M.va(i) = sum(M.fa(M.Rf{i}))/3;
%                 end
%                 toc;
% Old Vertex Angles:
% G = zeros(M.Nv,1);
% for i=1:M.Nf
%     tri = M.f(i,:);
%     %vertex indices of triangle points
%     idx_1 = tri(1);
%     idx_2 = tri(2);
%     idx_3 = tri(3);
%
%     %coordinates of triangle points
%     pt1 = M.v(idx_1,:);
%     pt2 = M.v(idx_2,:);
%     pt3 = M.v(idx_3,:);
%
%     %a_ij is the angle opposite to the edge connecting points i,j in the
%     %triangle
%     a12 = get_angle(pt1,pt2,pt3);
%     a23 = get_angle(pt2,pt3,pt1);
%     a31 = get_angle(pt3,pt1,pt2);
%
%     %G_ii holds the sum of the angles at vertex i in all triangles that
%     %contain vertex i
%     G(idx_1) = G(idx_1) + a23;
%     G(idx_2) = G(idx_2) + a31;
%     G(idx_3) = G(idx_3) + a12;
% end

% TODO: This piece of code extracts each row val into a cell array - but
% doesn't do it in order!
%         tic;
%         [I,~,values]=find(W);
%         u=unique([I,values],'rows');
%         N=size(u,1);
%         tmp=diff([0;find(diff(u(:,1)));N]);
%         W_rows=mat2cell(u(:,2),tmp);
%         % Finish the job
%         L = cell2mat(cellfun(@(Rv,Cot) sum(vmul(M.v(Rv,:),Cot)), M.Rv,W_rows ,'un',0))-M.v;
%         toc;

%         function [fi] = faces_from_vertices(M,vi)
%            fs = ismember(M.f,vi);
%            fi = find(sum(fs, 2) ~= 0);
%         end

assert(~(numel(fi) == M.Nf && size(edge_list,1)~=M.Ne));
% Edges may be computed in two ways - but the "face" way is
% stronger
warning('Edge list length (%d) does not match M.Ne (%d)',size(edge_list,1),M.Ne);
[I,J] = ind2sub([M.Nv,M.Nv],find(triu(M.A)));
ea_edge = [I,J];
if numel(ea_edge) > numel(edge_list)
    bigger_edge = ea_edge;
    smaller_edge = edge_list;
else
    bigger_edge = edge_list;
    smaller_edge = ea_edge;
end
edge_diff = setdiff(bigger_edge,smaller_edge,'rows')

function [DA,da] = dihedral_angles_adj(M,in_degs)
% Ref: www.grasshopper3d.com/forum/topics/convex-or-concave-angle-between-faces
if ~exist('in_degs','var'); in_degs = 0;end
M = M.add_face_normals();
fe = M.face_edges(); Nfe = size(fe,1);
da = zeros(Nfe,1);
for i=1:Nfe %TODO: Write vectoric implementation
    fi = fe(i,1); fj = fe(i,2);
    fni = M.fn(fi,:); fnj = M.fn(fj,:);
    ang = pi-atan2(normv(cross(fni, fnj, 2)), dot(fni, fnj, 2));
    
    % Check sign:
    w = M.fc(fi,:) - M.fc(fj,:); assert(any(round(w,8))); % Sanity
    if(round(dot(fni,w),8)>=0)
        % A convex polygon = All dihedral angles below 180
        assert(round(ang,8) <= pi); % Sanity
    else
        % A concave polygon = At least 1 dihedral angle above 180
        ang = 2*pi-ang;
        assert(round(ang,8) >= pi); % Sanity
    end
    if in_degs; ang = rad2deg(ang); end
    da(i) = ang;
end
DA = sparse([fe(:,1);fe(:,2)],[fe(:,2);fe(:,1)],[da;da],Nfe,Nfe);

end

%  for i = v_idxs
%       M.Rv{i} = find(M.A(i,:)~=0);
%  end

% M.A = logical(sparse(f, f(:,[2,3,1]), ones(nf,3), nv, nv));

function [K,H] = meyer_signed_curvature(M)
% The method is based on this paper:
% Meyer, M., Desbrun, M., Schröder, P., & Barr, A. H. (2003).
% Discrete differential-geometry operators for triangulated 2-manifolds.
% Notes 1: The curvatures data at the oundaries of domain is not reliable. It
% artificially gives zero instead of non-sense data.
%
% Notes 2:
% The above paper, gives a vector for MC and the absoulute value of mean
% curvature is half of the norm of this vector. Therfore, in this way the sign is
% not given. Here, I used a dot product betwen the MC vector and the normal
% vector at each point calculated based on weighted averaging of the
% triangle normal vectors given by MATLAB. I'm not sure what is the convention of
% MATLAB in determining the direction for the normal vectors. But it seems
% that they are all consistent toward one side of the surface. So the
% calculated signed MC shows the change of sign in data, but it can not
% gurantee that, for example, the positive MC is for a locally convex
% region (if The GC is positive).

M = M.add_face_areas();
M = M.add_face_normals();
fn = M.fn; fc = M.fc; fa = M.fa;
x = M.x(); y = M.y(); z = M.z(); f = M.f;

tri3d=triangulation(f,[x,y,z]);
bndry_edge=freeBoundary(tri3d);

l_edg = zeros(M.Nf,3);
v1 = zeros(M.Nf,3); v2 = zeros(M.Nf,3);  v3 = zeros(M.Nf,3);
ang_tri = zeros(M.Nf,3);
%%%% angles and edges of each triangle
for i=1:length(f(:,1))
    
    p1=f(i,1);
    p2=f(i,2);
    p3=f(i,3);
    
    v1(i,:)=[x(p2)-x(p1),y(p2)-y(p1),z(p2)-z(p1)];
    v2(i,:)=[x(p3)-x(p2),y(p3)-y(p2),z(p3)-z(p2)];
    v3(i,:)=[x(p1)-x(p3),y(p1)-y(p3),z(p1)-z(p3)];
    
    l_edg(i,1)=norm(v1(i,:));
    l_edg(i,2)=norm(v2(i,:));
    l_edg(i,3)=norm(v3(i,:));
    
    ang_tri(i,1)=acos(dot(v1(i,:)/l_edg(i,1),-v3(i,:)/l_edg(i,3)));
    ang_tri(i,2)=acos(dot(-v1(i,:)/l_edg(i,1),v2(i,:)/l_edg(i,2)));
    ang_tri(i,3)=pi-(ang_tri(i,1)+ang_tri(i,2));
    
end
a_mixed=zeros(1,length(x));
alf=zeros(1,length(x));
K=zeros(length(x),1);
H=zeros(length(x),1);
for i=1:length(x)
    mc_vec=[0,0,0];
    n_vec=[0,0,0];
    
    if ~isempty(bndry_edge) && ~isempty((find(bndry_edge(:,1)==i)))
    else
        clear neib_tri
        neib_tri=vertexAttachments(tri3d,i);
        
        for j=1:length(neib_tri{1})
            neib=neib_tri{1}(j);
            
            %%%% sum of angles around point i ===> GC
            for k=1:3
                if f(neib,k)==i
                    alf(i)=alf(i)+ ang_tri(neib,k);
                    break;
                end
            end
            
            %%%%% mean curvature operator
            if     k==1
                mc_vec=mc_vec+(v1(neib,:)/tan(ang_tri(neib,3))-v3(neib,:)/tan(ang_tri(neib,2)));
            elseif k==2
                mc_vec=mc_vec+(v2(neib,:)/tan(ang_tri(neib,1))-v1(neib,:)/tan(ang_tri(neib,3)));
            elseif k==3
                mc_vec=mc_vec+(v3(neib,:)/tan(ang_tri(neib,2))-v2(neib,:)/tan(ang_tri(neib,1)));
            end
            
            
            %%% A_mixed calculation
            if(ang_tri(neib,k)>=pi/2)
                a_mixed(i)=a_mixed(i)+fa(neib)/2;
            else
                if (any(ang_tri(neib,:)>=pi/2))
                    a_mixed(i)=a_mixed(i)+fa(neib)/4;
                else
                    sum=0;
                    for m=1:3
                        if m~=k
                            ll=m+1;
                            if ll==4       %% p1==>l2   ,p2==>l3   ,p3==>l1
                                ll=1;
                            end
                            sum=sum+(l_edg(neib,ll)^2/tan(ang_tri(neib,m)));
                        end
                    end
                    a_mixed(i)=a_mixed(i)+sum/8;
                end
            end
            
            %%%% normal vector at each vertex
            %%%% weighted average of normal vecotors of neighbour triangles
            wi=1/norm([fc(neib,1)-x(i),fc(neib,2)-y(i),fc(neib,3)-z(i)]);
            n_vec=n_vec+wi*fn(neib,:);
            
        end
        
        K(i)=(2*pi()-alf(i))/a_mixed(i);
        
        mc_vec=0.25*mc_vec/a_mixed(i);
        n_vec=n_vec/norm(n_vec);
        %%%% sign of MC
        if dot(mc_vec,n_vec) <0
            H(i)=-norm(mc_vec);
        else
            H(i)=norm(mc_vec);
        end
        
    end
end
end


