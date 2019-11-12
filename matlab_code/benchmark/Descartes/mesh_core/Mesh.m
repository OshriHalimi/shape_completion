classdef Mesh
    properties
        path % Origin
        file_name
        f % Faces
        v % Vertices
        Ne % Number of edges
        Nv % Number of vertices
        Nf % Number of faces
        name % Name of the mesh
        plt % Plotting tools structure
        
        A % Adjacency Matrix
        
        % Optional:
        Rv % Ring Vertex
        Rf % Ring Tri
        d % Degree vector
        D % Degree Matrix
        
        fn % Face Normals
        fc % Face Centeroids
        fcc % Face Voronoi Centers
        vn % Vertex Normals
        fa % Face Areas
        va % Vertex Areas
        
        % Collaterals
        FE2E 
        E2FE
        fe
        e_feal
        Nfe
        AF
    end
    methods
%-------------------------------------------------------------------------%
%                            Constructor
%-------------------------------------------------------------------------%
        function [M]= Mesh(v,f,name,pth)
            
            % Input Handle: 
            if nargin ==1 % Presume FV input
                f = v.faces;
                v = v.vertices;
            end  
            if nargin < 3 ; name = ''; end
            if nargin <4; pth = ''; end

            if(size(v,1)==3); v = v.';end
            if(size(f,1)==3); f = f.'; end
            assert(size(v,2)==3 && (size(f,2)==3 || isempty(f)),'Mesh class works with tris only');
            M.name = name; M.f = f; M.v = v; M.path = pth; 
            
            % Compute other fields: 
            if ~isempty(pth)
                [~,file_name,ext] = fileparts(pth);
                M.file_name = [file_name,ext]; 
            end
            M.Nv = size(v,1);
            M.Nf = size(f,1);
            
            % Compute adjacency matrix
%             fv = M.vertex_face_map(); B = logical(fv*fv.');
%             M.A = logical(B-diag(diag(B)));
%             % assert(logical(issymmetric(M.A)));
%             % Compute number of edges
%             M.Ne = nnz(M.A)/2;
%             % Plotting Options
            M = Mesh_plot_params(M);
        end
%-------------------------------------------------------------------------%
%                              Elementry Ops
%-------------------------------------------------------------------------%
        function export_mat(M,name)
            if ~exist('name','var')
                % Prep name with no spaces: 
                name = [M.file_name '.mat'];
            else
                [~,name] = fileparts(name);
                name = [name '.mat'];
            end
            save(name,'M');
        end
        function export_as(M,name)
            write_mesh(M,name);
        end
        function [FV] = fv_struct(M)
            FV.faces = M.f;
            FV.vertices = M.v;
        end
        function export_html(M,Cv,rotate)
            % TODO : Rewrite mesh2html
            if ~exist('Cv','var') || isempty(Cv)
                Cv = uniclr('w',M.Nv); 
            else
                assert(size(Cv,1)==M.Nv); 
            end
            if ~exist('rotate','var'); rotate = 0; end
            mesh2html(M.f,M.v,1,'name',M.name,'subheading',M.plt.title,'color',Cv,'rotation',rotate);
        end
%-------------------------------------------------------------------------%
%                                Visulization
%-------------------------------------------------------------------------%
        function [h]= visualize_heatmap(M,C,plt_override)
            M.plt.light = 0;
            %             M.plt.clr_map.trunc= 'sym';
            if ~exist('C','var'); C = uniclr('c',M.Nf); end
            if ~ exist('plt_override','var'); plt_override = struct(); end
            h = ezvisualize(M,C,plt_override);
        end
        function [h] = ezvisualize(M,C,plt_override)
            if ~exist('C','var') || isempty(C)
                C = uniclr('w',M.Nv);
                M.plt.S.FaceColor = 'flat';
            elseif ischar(C)
                C = uniclr(C,M.Nv); 
                M.plt.S.FaceColor = 'flat';
            elseif size(C,1) == M.Nv
                M.plt.S.FaceColor = 'interp';
                M.plt.clr_bar = 0;
            else
                M.plt.S.FaceColor = 'flat';
                M.plt.clr_bar = 0;
            end
            M.plt.S.FaceVertexCData = C;
            if exist('plt_override','var')
                M.plt = mergestruct(M.plt,plt_override,1);
            end
            h = Mesh_visualize(M);
        end
        function [h] = visualize(M,C,plt_override)
            if nargin>1 && ~isempty(C)
                M.plt.S.FaceVertexCData = C;
                M.plt.clr_bar = 1; %Doesn't change M outside the function
                if(size(C,1)==M.Nv) %Vertex Colores
                    M.plt.S.FaceColor = 'interp';
                    M.plt.S.Marker = 'o';
                    M.plt.S.EdgeColor = [0,0,0];
                    M.plt.S.LineWidth = 1;
                    M.plt.S.LineStyle = '-';
                    M.plt.S.MarkerFaceColor = 'flat';
                else % Face Colors
                    M.plt.S.FaceColor = 'flat';
                    M.plt.S.EdgeColor = [1,1,1];
                    M.plt.S.LineWidth = 1;
                    M.plt.S.LineStyle = '-';
                end
            else % No Color Input
                M.plt.S.FaceVertexCData = M.v(:,2); % Orient colors to Y
                M.plt.S.FaceColor = 'interp'; % Interpolate face colors from verts
            end
            if exist('plt_override','var')
                M.plt = mergestruct(M.plt,plt_override);
            end
            h = Mesh_visualize(M);
        end
        function wireframe(M,C,plt_override)
            M.plt.S.FaceColor = 'none';
            if exist('C','var') && ~isempty(C)
                M.plt.S.EdgeColor = C;
            else
                M.plt.S.EdgeColor = [0,0,0];
            end
            M.plt.S.LineWidth = 1;
            M.plt.S.LineStyle = '-';
            if exist('plt_override','var')
                M.plt = mergestruct(M.plt,plt_override);
            end
            Mesh_visualize(M);
        end
        function visualize_vertices(M,C,plt_override)
            M.plt.S.FaceColor = 'none';
            M.plt.S.Marker = 'o';
            if exist('C','var') && ~isempty(C)
                assert(size(C,1)==M.Nv);
                M.plt.S.FaceVertexCData = C;
%                 M.plt.clr_bar = 1;
            else
                M.plt.S.FaceVertexCData = M.v(:,2); % Orient colors to Y
            end
            M.plt.S.MarkerFaceColor = 'flat';
            if exist('plt_override','var')
                M.plt = mergestruct(M.plt,plt_override,1);
            end
            Mesh_visualize(M);
        end
        function visualize_ve_map(M,C,fclr)
            if ~exist('fclr','var'); fclr = [1,1,1]; end 
            assert(size(C,1)==M.Nv); 
            h = M.visualize(C);
            h.EdgeColor = 'interp'; h.FaceColor = fclr; 
        end
        function status(M)
            mesh_status(M); 
        end
%-------------------------------------------------------------------------%
%                                Topology
%-------------------------------------------------------------------------%
        function [orient_two_manifold] = is_oriented_2manifold(M)
            orient_two_manifold = M.is_2manifold() && M.is_oriented();
        end
        function [is_cmplt] = is_complete(M)
            cncomp = conn_comp(M.A);
            is_cmplt = (cncomp==1); 
        end
        function [watertight] = is_watertight(M) % No Border
            watertight = isempty(M.boundary());
        end
        function [two_manifold] = is_2manifold(M) % No Singularties
            two_manifold = isempty(M.singularity());
        end
        function [convex] = is_convex(M)
            [~,~,is_concave,~] = M.dihedral_angles_adj(); 
            convex = any(is_concave); 
        end
        function [oriented] = is_oriented(M)
            % TODO - Find some better way to compute this - too expensive
            [~,flipped] = orient_mesh(M.v,M.f);
            oriented = (sum(flipped)==0);
        end
        function [euler_chr] = euler_char(M)
            assert(M.is_watertight(),'Manifold is not watertight');
            euler_chr = M.Nv - M.Ne + M.Nf;
        end
        function [gen] = genus(M)
            gen = 1-M.euler_char()/2;
        end
        function [lavg,lmin,lmax,ltot] = edge_len_statistics(M)
            Di = M.edge_distances();
            lavg =mean(Di);
            lmin = min(Di);
            lmax = max(Di);
            ltot = sum(Di);
        end
        function [vavg,vmin,vmax,vtot] = vertex_valence_statistics(M)
            M = M.add_degree_vec();
            vavg =mean(M.d);
            vmin = min(M.d);
            vmax = max(M.d);
            vtot = sum(M.d);
        end
        function [ AR ] = area( M )
            M = M.add_face_areas();
            AR = sum(M.fa);
        end
        function [V] = volume(M)
            % TODO - Find better implementation 
            % assert(M.is_watertight(),'Manifold is not watertight');
            V = 0;
            v_centered = bsxfun(@minus, M.v, mean(M.v,1)); % Center Vs
            for i = 1:M.Nf
                % volume of current tetrahedron
                V =  V + det(v_centered(M.f(i, :), :)) / 6;
            end
        end
        function [B,R,D] = box_bounds(M)
            B= [min(M.v) ; max(M.v) ; 0.5*(min(M.v) + max(M.v))];
            R = normv(B(1,:)-B(2,:));
            D = R*2;
        end
        function [v_ang,ang_mat] = angle_defect(M)
            % Find orig edge lengths and angles
            L1 = normv(M.v(M.f(:,2),:)-M.v(M.f(:,3),:));
            L2 = normv(M.v(M.f(:,1),:)-M.v(M.f(:,3),:));
            L3 = normv(M.v(M.f(:,1),:)-M.v(M.f(:,2),:));
            
            A1 = (L2.^2 + L3.^2 - L1.^2) ./ (2.*L2.*L3);
            A2 = (L1.^2 + L3.^2 - L2.^2) ./ (2.*L1.*L3);
            A3 = (L1.^2 + L2.^2 - L3.^2) ./ (2.*L1.*L2);
            AA = acos([A2,A3,A1]); % The angles of each face in radians
            ang_mat = sparse(M.f, M.f(:,[2,3,1]), AA, M.Nv, M.Nv);
            v_ang = full(sum(ang_mat,1)).';
        end
        function [ff2]= fv2vf(M,ff1)
            if size(ff1,1) == M.Nf
                [f2v,~] = face_vertex_interpolation(M);
                ff2 = f2v*ff1;
            elseif size(ff1,1) == M.Nv
                [~,v2f] = face_vertex_interpolation(M);
                ff2 = v2f*ff1;
            else
                error('Invalid ff1 size'); 
            end
        end
%-------------------------------------------------------------------------%
%                                  Edges
%-------------------------------------------------------------------------%
        function [DA,da,is_concave,fe] = dihedral_angles_adj(M,in_degs,km)
            % Ref: www.grasshopper3d.com/forum/topics/convex-or-concave-angle-between-faces
            if ~exist('in_degs','var'); in_degs = 0;end
            if ~exist('km','var'); km = 0;end

            M = M.add_face_normals(); fe = M.face_edges(); 
            fi = fe(:,1); fj = fe(:,2); 
            fni = M.fn(fi,:); fnj = M.fn(fj,:);
            % These are the unsigned dehedral angles: 
            da = pi-atan2(normv(cross(fni, fnj, 2)), dot(fni, fnj, 2)); 
            % da_deg = rad2deg(da); 
            % Determine sign: 
            w = M.fc(fi,:) - M.fc(fj,:); 
            is_concave = round(dot(fni,w,2),8)<0; 
            da(is_concave) = 2*pi-da(is_concave); 
            if in_degs; da = da*180/pi; end

            if km
                [id,C] = kmeans(da,km); % TODO - Find better way
                for i=1:km
                    da(id==i) = C(i); 
                end
            end
            DA = sparse([fe(:,1);fe(:,2)],[fe(:,2);fe(:,1)],[da;da],M.Nf,M.Nf); 
        end
        function [e,ei] = fe2e(M,fei)
            M = M.add_fedge_set();
            ei = M.FE2E(fei);
            e = M.e_feal(fei,:); 
        end
        function [fe,fei] = e2fe(M,ei)
            M = M.add_fedge_set();
            fei = M.E2FE(ei);
            fe = M.fe(fei,:); 
        end
        function [FE2E,E2FE,fe,e_aligned,AF] = fedge2edge_tables(M)
            [fe,AF] = M.face_edges();
            EF = M.edge_face_map(1);
            FE2E = zeros(size(fe,1),1); 
            E2FE = zeros(size(fe,1),1); 
            for i=1:size(fe,1) % TODO; Remove loop & Find a more stright foward connection fe->e 
                ei= find(EF(:,fe(i,1)) & EF(:,fe(i,2)));
                if numel(ei)~= 1
                    error('Identical faces found'); %TODO - Write dedup to fix this
                end
                FE2E(i) = ei; 
                E2FE(ei) = i; 
            end
            e_aligned = M.edges(); 
            e_aligned = e_aligned(FE2E,:); 
            % e = M.edges(); etp = 1:5; 
            % fi = fe(etp,:); fi = fi(:); 
            % M.ezvisualize(uniclr('gold',M.Nf,'w',fi));
            % add_edge_visualization(M,fe(etp,:),1,'b'); 
            % add_edge_visualization(M,e(EF2E(etp),:),0,'r'); 
        end
        function [fe,AF] = face_edges(M)
            if isempty(M.fe)
                AF = M.face_adj();
                [I,J] = ind2sub([M.Nf,M.Nf],find(triu(AF)));
                fe = [I,J];
            else
                fe = M.fe; 
                AF = M.AF; 
            end
        end
        function [edge_list,fe2ue_map,ue2fe_map] = edges(M,fi)
            if ~exist('fi','var'); fi = 1:M.Nf;end
            fs = M.f(fi,:);
            ex_edge = sort([fs(:,[1 2]) ; fs(:,[2 3]) ; fs(:,[3 1])], 2);
            [edge_list,fe2ue_map,ue2fe_map] = unique(ex_edge, 'rows');
            
%             if(numel(fi) == M.Nf && size(edge_list,1)~=M.Ne)
%                 % Edges may be computed in two ways - but the "face" way is
%                 % stronger
%                 warning('Edge list length (%d) does not match M.Ne (%d)',size(edge_list,1),M.Ne);
%                 [I,J] = ind2sub([M.Nv,M.Nv],find(triu(M.A)));
%                 ea_edge = [I,J];
%                 if numel(ea_edge) > numel(edge_list)
%                     bigger_edge = ea_edge;
%                     smaller_edge = edge_list;
%                 else
%                     bigger_edge = edge_list;
%                     smaller_edge = ea_edge;
%                 end
%                 edge_diff = setdiff(bigger_edge,smaller_edge,'rows')
%             end
        end
        function [D,E] = edge_distances(M,use_face_edges)
            if ~exist('use_face_edges','var'); use_face_edges=0; end
            if use_face_edges
                M = M.add_face_centers(); 
                E = M.face_edges();
                D = vecnorm(M.fc(E(:,1),:)-M.fc(E(:,2),:),2,2);
            else
                E = M.edges();
                D = vecnorm(M.v(E(:,1),:)-M.v(E(:,2),:),2,2);
            end
        end
        function [me,mv,mf,mei] = manifold(M,varargin)
            [me,mv,mf,mei] = edge_classification_(M,'2-manifold',varargin);
        end
        function [se,sv,sf,sei] = singularity(M,varargin)
            [se,sv,sf,sei] = edge_classification_(M,'singular',varargin);
        end
        function [be,bv,bf,bei] = boundary(M,varargin)
            [be,bv,bf,bei] = edge_classification_(M,'boundary',varargin);
        end
        
        function [fe] = feature_edges(M,theta,to_show)
            %A feature edge is:
            %A boundary or singular edge
            %Or shared by a pair of triangles with angular deviation greater than the angle theta.
            x= M.x(); y = M.y(); z = M.z();
            tr = triangulation(M.f, x,y,z);
            fe = featureEdges(tr,theta);
            if exist('to_show','var') && to_show
                M.plt.title = sprintf('Feature edges for \\theta=%g Rad',theta);
                M.wireframe(); add_edge_visualization(M,fe);
            end
        end
        function [E2] = long_edges(M,len,to_show)
            if ~isscalar(len)
                len1 = len(1); len2 = len(2);
            else
                len1 = len; len2 = Inf;
            end
            [Dist,E] = edge_distances(M);
            E2 = E(Dist>=len1 & Dist<=len2,:);
            if exist('to_show','var') && to_show
                M.plt.title = sprintf('Edge with length\\in[%g,%g]',len1,len2);
                M.wireframe(); add_edge_visualization(M,E2);
            end
        end
        function [ RE1, RE2, RE3 ] = rot_face_edge_vectors( M )
            [E1,E2,E3] = M.face_edge_vectors();
            M = M.add_face_normals();
            RE1 = cross(M.fn, E1 ); % Rotate the vectors
            RE2 = cross(M.fn, E2 );
            RE3 = cross(M.fn, E3 );
        end
        function [ E1, E2, E3 ] = face_edge_vectors( M )
            E1 = M.v(M.f(:,3),:) - M.v(M.f(:,2),:);
            E2 = M.v(M.f(:,1),:) - M.v(M.f(:,3),:);
            E3 = M.v(M.f(:,2),:) - M.v(M.f(:,1),:);
        end
%-------------------------------------------------------------------------%
%                        Secondary Data Structures
%-------------------------------------------------------------------------%
        function [nei] = neighbors(M,I,type,inclusive,n)
            if ~exist('inclusive','var'); inclusive = 0; end
            if ~exist('n','var'); n = 1; end
            switch type
                case 'vv'
                    Adj = M.A+speye(M.Nv);
                case 'ff'
                    Adj = M.face_adj()+speye(M.Nf);
                case 'vf'
                    Adj = M.vertex_face_map();
                case 'fv'
                    Adj = M.vertex_face_map().';
            end
            
            for i=1:n
                if size(Adj,1) == size(Adj,2) % Continue the same way
                    [~,col_ids,~]=find(Adj(I,:));
                    nei = unique(col_ids);
                    if i==n && ~inclusive % Last iteration
                        nei = setdiff(nei,I);
                    else
                        I = nei; % Set the next set
                    end
                else % Only done once
                    [~,col_ids,~]=find(Adj(I,:));
                    nei = unique(col_ids);
                    if n>1
                        nei = M.neighbors(nei,[type(2),type(2)],inclusive,n-1);
                    end
                end
            end
        end
        function [AF] = face_adj(M,with_boundry,manifold_only)
            % Does not use the M.AF due to configurations 
            if ~exist('manifold_only','var'); manifold_only = 0; end
            if ~exist('with_boundry','var'); with_boundry = 0; end
            % If manifold_only, we will hold only the boundary & 2manifold
            % edges
            EF = M.edge_face_map(manifold_only);
            AF = EF'*EF;
            if with_boundry
                % The Dual graph sometimes defines an additional vertex that
                % acts as the "boundary" - Let's add it if requested 
                be = sum(EF,2)==1; 
                bf_row = [sum(EF(be,:)),0]; % 2 here mean: 2/3 edges of the face are boundary
                AF(M.Nf+1,1:M.Nf+1) = bf_row; AF(1:M.Nf+1,M.Nf+1); 
            end
            AF = logical(AF-diag(diag(AF)));
            
        end
        function [EF] = edge_face_map(M,manifold_only)
            if ~exist('manifold_only','var'); manifold_only = 0; end
            [~,~,ue2fe_map] = M.edges();
            EF = sparse(ue2fe_map(:),repmat(1:M.Nf,1,3)',1);
            % Kill Non-manifold edges:
            if manifold_only
                EF(sum(EF,2)>2,:) = 0;
            end
        end
        function [ve] = vertex_edge_map(M)
            % This is also known as the "incidence" matrix
            E = M.edges();
            ve = sparse(E,repmat((1:M.Ne).',1,2),ones(M.Ne,2));
        end
        function [vf] = vertex_face_map(M)
            vf=sparse(M.f,repmat((1:M.Nf).',1,3),ones(M.Nf,3),M.Nv,M.Nf);
        end
        function [X] = x(M,vs)
            if nargin > 1; X = M.v(vs,1);else; X = M.v(:,1);end
        end
        function [Y] = y(M,vs)
            if nargin > 1; Y = M.v(vs,2); else; Y = M.v(:,2);end
        end
        function [Z] = z(M,vs)
            if nargin > 1; Z = M.v(vs,3); else; Z = M.v(:,3);end
        end
%-------------------------------------------------------------------------%
%                           Data Structure Mutators
%-------------------------------------------------------------------------%
        function [M]=add_face_ring(M)
            if isempty(M.Rf)
                M.Rf = cell(M.Nv,1); %TODO - Write faster implementation
                for i = 1:M.Nv
                    M.Rf{i} = find(sum(ismember(M.f,i),2));
                end
            end
        end
        function [M]=add_vertex_ring(M)
            if isempty(M.Rv)
                M.Rv = cell(M.Nv,1);
                % This piece of code extracts the non-zero elements from
                % each row of the matrix M.A
                [c, r] = find((M.A).');
                M.Rv = accumarray(r, c, [size((M.A),1), 1], @(L) {L.'} );
            end
        end
        function [M]= add_degree_vec(M)
            if isempty(M.d)
                M.d = full(sum(M.A,1));
            end
        end
        function [M]=add_degree_mat(M)
            if isempty(M.D)
                M.D = diag(sum(M.A,1));
            end
        end
        function [M]=add_face_normals(M)
            % TODO: Check if outward/inward || 0 normals
            if isempty(M.fn) 
                % Face Normals
                a = M.v(M.f(:,1),:);
                b = M.v(M.f(:,2),:);
                c = M.v(M.f(:,3),:);
                M.fn = cross((b-a),(c-a)); % un-normalized face normals
                M.fn = M.fn./repmat(sqrt(sum(M.fn.^2,2)),[1,3]); % normalized face normals
                % Face Centeroids
            end
            M = M.add_face_centers(); 

        end
        function [M] = add_face_centers(M)
            if isempty(M.fc)
                M.fc = (M.v(M.f(:,1),:)+M.v(M.f(:,2),:)+M.v(M.f(:,3),:))./3;
            end
        end
        function [M]=add_vertex_normals(M)
            if isempty(M.vn)
                tic;
                M = M.add_face_normals();
                M.vn = zeros(M.Nv,3);
                M.vn = accumarray([M.f(:),ones(size(M.f(:)));M.f(:),...
                    2*ones(size(M.f(:))); M.f(:),3*ones(size(M.f(:)))],...
                    [repmat(M.fn(:,1),[3,1]);repmat(M.fn(:,2),[3,1]);repmat(M.fn(:,3),[3,1])]);
                M.vn = M.vn./repmat(sqrt(sum(M.vn.^2,2)),[1,3]); % Normalize
                
            end
        end
        function [M] = add_face_areas(M)
            if isempty(M.fa)
                v1 = M.v(M.f(:,1), :);
                v12 = M.v(M.f(:,2), :) - v1;
                v13 = M.v(M.f(:,3), :) - v1;
                M.fa = vecnorm((cross(v12, v13)),2,2)/2;
            end
        end
        function [M] = add_vertex_areas(M)
            if isempty(M.va)
                M = M.add_face_areas();
                V = [M.fa;M.fa;M.fa;];
                vf_areas = sparse(M.f, M.f(:,[2,3,1]), V, M.Nv,M.Nv); %CHECK this
                M.va = full((sum(vf_areas, 2)) / 3);
            end
        end
        function [M] = add_circumcenters(M)
            if isempty(M.fcc)
                M.fcc = circumcenter(triangulation(M.f,M.v));
                % TODO: Add Voronoi Areas
                % Radius of circle is just the distance of any point in the face
                % to its circumcenter
            end
        end
        function [M] = add_fedge_set(M)
            if isempty(M.fe)
                [M.FE2E,M.E2FE,M.fe,M.e_feal,M.AF] = M.fedge2edge_tables();
                M.Nfe = size(M.fe,1); 
            end
            M = M.add_face_centers(); 
        end
    end
end % END CLASS

%-------------------------------------------------------------------------%
%                            Private Functions
%-------------------------------------------------------------------------
function [e,v,f,ei] = edge_classification_(M,type,args)
if length(args)<1 || isempty(args{1})
    fi = 1:M.Nf;
else
    fi = args{1};
end
if length(args)<2
    to_show = 0;
else
    to_show = args{2};
end
[edge_list,~,ue2fe_map] = M.edges(fi);
counts=countmember(ue2fe_map,ue2fe_map);

switch type
    case 'boundary' %boundary
        counts = (counts ==1);
    case '2-manifold' % 2manifold
        counts = (counts ==2);
    case 'singular' %singularties
        counts = (counts >2);
    otherwise
        error('Incorrect type');
end 
% TODO: Check if Face-Edge is faster to compute
e = edge_list(unique(ue2fe_map(counts)),:);
ei = find(ismember(M.edges(),e,'rows')); 
v = unique(e(:));
f = mod(find(counts),M.Nf);
f(f==0)=M.Nf;
if exist('to_show','var') && to_show %TODO - insert support for fi
    oplt.clr_bar =0;
    M.plt.title=usprintf('%s Edges : %d',type,numel(e));
    M.wireframe(); add_edge_visualization(M,e);
    M.plt.title=usprintf('%s Faces : %d',type,numel(f));
    M.visualize(uniclr('t',M.Nf,'r',f),oplt);
    M.plt.title=usprintf('%s Vertices : %d',type,numel(v));
    M.visualize(uniclr('t',M.Nv,'r',v),oplt);
end
end

%-------------------------------------------------------------------------%
%                        Some Documentation
%-------------------------------------------------------------------------%
%   The volume is computed as the sum of the signed volumes of tetrahedra
%   formed by triangular faces and the centroid of the mesh. Faces need to
%   be oriented such that normal points outwards the mesh. See:
%   http://stackoverflow.com/questions/1838401/general-formula-to-calculate-polyhedron-volume
