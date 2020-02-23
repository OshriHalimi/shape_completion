clearvars; close all; clc; test_setup(); cd(up_script_dir(0));
%-------------------------------------------------------------------------%
%
%-------------------------------------------------------------------------%
[M,libm ,menu] = test_mesh(3,'small');
%-------------------------------------------------------------------------%
%                         Internal Mesh Class Tests
%-------------------------------------------------------------------------%
% visualization_test(M);
% topology_test(M);
% edges_test(M);
% misc_test(M);
% areas_test(M);
% normals_test(M);
% dihedral_angles_test(M);
%-------------------------------------------------------------------------%
%                         External Mesh Class Tests
%-------------------------------------------------------------------------%
% plot_pair_test(M);
curvature_test(M);
% discrete_ops_test(M);
% IO_test(M);
% bound_box_test(M);
fprintf('-I- Test Completed\n');
%-------------------------------------------------------------------------%
%
%-------------------------------------------------------------------------%
function visualization_test(M)

M.visualize(uniclr('teal',M.Nf));
M.visualize(uniclr('gold',M.Nf));
[cd,c] = mesh_centeroid_dist(M);
M.visualize(cd);add_feature_pts(M,c,0.2);
M.visualize_ve_map(cd,'w'); add_feature_pts(M,c,0.1);

add_vertex_labels(M);
M.visualize(uniclr('tot_rand_red',M.Nf));
M.visualize(uniclr('tot_rand',M.Nv));
M.visualize(uniclr('c',M.Nf));
M.visualize(uniclr('tot_rand',M.Nf));

M.ezvisualize();
M.ezvisualize(uniclr('tot_rand',M.Nv));
M.ezvisualize(uniclr('b',M.Nv));
M.ezvisualize(uniclr('r',M.Nf));
M.ezvisualize(uniclr('tot_rand',M.Nf));

% M.plt.clr_map.trunc = 'sym';
% M.visualize_heatmap();
M.plt.clr_map.name = 'magma';
M.plt.clr_map.trunc = 'none';
M.visualize_heatmap();
M.visualize_heatmap(cd);
ovplt.S.EdgeColor = [1,1,1];
ovplt.S.LineStyle = '-';
M.visualize_heatmap(cd,ovplt);

M.wireframe();
M.visualize_vertices();
M.visualize_vertices(uniclr('tot_rand',M.Nv));
add_vertex_labels(M);
M.visualize_vertices(uniclr('w',M.Nv));
pts = M.v(randperm(M.Nv,20),:);
add_feature_pts(M,pts,0.05);

end
function topology_test(M)
is_watertight=M.is_watertight()
is_2manifold=M.is_2manifold()
genus=M.genus()
euler_char=M.euler_char()
[vavg,vmin,vmax,vtot]=M.vertex_valence_statistics()
[lavg,lmin,lmax,ltot] = M.edge_len_statistics()
area=M.area()
volume=M.volume()
[~,~,h] = minboundbox(M.x(),M.y(),M.z(),'v',1);
h
bbbox_avg = mean(h)
mb = mean_breadth(M)
[B,R,D] = M.box_bounds()
[sqrt(h(1)^2+h(2)^2),sqrt(h(2)^2+h(3)^2),sqrt(h(3)^2+h(1)^2)]
end

function edges_test(M)
M.singularity();
[se,sv,sf] = M.singularity(1);
M.boundary();
[be,bv,bf] = M.boundary([],1);
M.feature_edges(pi/4);
[fe] = M.feature_edges(pi/4,1)

end
function misc_test(M)
M.vertex_edge_map();
M.vertex_face_map();
MM = M.mass_matrix();
M.vertex_angle_sum();
end
function areas_test(M)
M = M.add_vertex_areas();
M.visualize(M.va); title(sprintf('%s colored by vertex areas',M.name'));
M = M.add_face_areas();
M.visualize(M.fa); title(sprintf('%s colored by face areas',M.name'));
end
function dihedral_angles_test(M)
THETA = 0.2;
[DA] = M.dihedral_angles_adj();
% Adjacency matrix of nearly coplanar faces:
DA_coplanar = DA>=(pi-THETA);
[~,C] = conn_comp(DA_coplanar);
M.plt.title = sprintf('Segmentation by dihedral angles with \\theta = %g',THETA);
M.ezvisualize(C.');

CM = jet(numel(unique(C)));
colormap(shuffle(CM));
M.plt.title = 'Maximum dihedral angle of each face';
M.ezvisualize(full(max(DA,[],2)));

% HACKY: Make minimum angles most negative
ii = find(DA);
for k=1:length(ii)
    DA(ii(k)) = DA(ii(k)) - 2*pi;
end
M.plt.title = 'Minimum dihedral angle of each face';
M.ezvisualize(full(min(DA,[],2)));
end
function normals_test(M)
% ov.normal_scaling = 3;
o.pick = 0;

M = M.add_face_normals();
M = M.add_vertex_normals();

M.ezvisualize(uniclr('w',M.Nv)); add_vectorfield(M,M.fn,o);
title('Face Normals on clean surface');

M.visualize_vertices(uniclr('w',M.Nv)); add_vectorfield(M,M.vn,o);
title('Vertex Normals on clean surface');

M.ezvisualize(M.fn); add_vectorfield(M,M.fn,o);
title('Face Normals with Vectorfield coloring');

M.ezvisualize(M.vn); add_vectorfield(M,M.vn,o);
title('Vertex Normals with Vectorfield coloring');

M.wireframe(); add_vectorfield(M,M.fn,o); add_vectorfield(M,M.vn,o);
legend({sprintf('%s edges',M.name),'Face Normals','Vertex Normals'});
end
%-------------------------------------------------------------------------%
%
%-------------------------------------------------------------------------%
function plot_pair_test(M)
% M1 = test_mesh(3,'small');
% M2 = test_mesh(4,'small');
mesh_plot_pair(M,M,uniclr('w',M.Nf,'k',1:2:M.Nf),uniclr('w',M.Nf,'c',1:2:M.Nf));
end
function IO_test(M)
M.export_mat('Booboo.mat');
M.export_mat('Booboo');
M.export_mat();
read_mesh('Booboo.mat');

M.export_as('Hello.mat');
M.export_as('Hello.off');
M.export_as('Hello.obj');
M.export_as('Hello.ply');
M.export_as('Hello.smf');
M.export_as('Hello.wrl');

read_mesh('Hello.off');
read_mesh('Hello.obj');
read_mesh('Hello.ply');
read_mesh('Hello.smf');
read_mesh('Hello.wrl');
fileparts(M.path);
delete('Booboo.mat', 'Hello.mat','Hello.off','Hello.obj','Hello.ply','Hello.smf','Hello.wrl'...
    ,[M.file_name '.mat']);
end

function bound_box_test(M)
M.visualize();
add_bounding_box(M);
end
function curvature_test(M)
[K_mu,H_mu]=visualize_curvature(M,'meyer'); % This is the fastest
[K_ms,H_ms]=visualize_curvature(M,'szymon'); %The only one that gets the signed of the mean curvature right
[K_s,H_s] =visualize_curvature(M,'dirk_jan');
[K_dj,H_dj] =visualize_curvature(M,'high_dirk_jan');
end
function discrete_ops_test(M)
[ D,G,M ] = mesh_divergence(M);
oplt.normal_clr = 'w';
% oplt.clr_map.name = 'parula';
% oplt.normal_scaling = 0.05;
o.pick = 0.1;

% gradient_plot(M,D,G,oplt,o,M.v(:,1),'X Axis');
% gradient_plot(M,D,G,oplt,o,M.v(:,2),'Y Axis');
% gradient_plot(M,D,G,oplt,o,M.v(:,3),'Z Axis');

gradient_plot(M,D,G,oplt,o,M.va,'Vertex Area');
gradient_plot(M,D,G,oplt,o,mesh_centeroid_dist(M),'Distance from Centeroid');
% [K,absH] = mesh_curvature(M,'exact');
% gradient_plot(M,D,G,oplt,o,K,'Gaussian Curvature');
% gradient_plot(M,D,G,oplt,o,absH,'Mean Curvature ');

% vfx = repmat( [1 0 0], M.Nf, 1 ); vfy = repmat( [0 1 0], M.Nf, 1 );
% vfz = repmat( [0 0 1], M.Nf, 1 ); vfxy = repmat( [1,1,0], M.Nf, 1 );

% divergence_plot(M,D,oplt,o,vfx,'Projected Uniform Field in the X direction');
% divergence_plot(M,D,oplt,o,vfy,'Projected Uniform Field in the Y direction');
% divergence_plot(M,D,oplt,o,vfz,'Projected Uniform Field in the Z direction');
% divergence_plot(M,D,oplt,o,vfxy,'Projected Uniform Field in the XY direction');
end
%-------------------------------------------------------------------------%
%
%-------------------------------------------------------------------------%
function gradient_plot(M,D,G,oplt,o,f,fname)
o.normalize = 0;
M.plt.title = sprintf('%s function & its gradient',fname); M.ezvisualize(f,oplt);
add_vectorfield(M,G*f,o,oplt);
o.normalize = 1;
M.plt.title = sprintf('%s function & its normalized gradient',fname); M.ezvisualize(f,oplt);
add_vectorfield(M,G*f,o,oplt);

o.normalize = 0;
M.plt.title = sprintf('Gradient with Laplacian coloring of the %s function',fname); M.ezvisualize(D*G*f,oplt);
add_vectorfield(M,G*f,o,oplt);
o.normalize = 1;
M.plt.title = sprintf('Normalized gradient with Laplacian coloring of the %s function',fname); M.ezvisualize(D*G*f,oplt);
add_vectorfield(M,G*f,o,oplt);

M.plt.title = sprintf('Gradient with Scaled Vectorfield coloring of the %s function',fname); M.ezvisualize(reshape(mat2gray(G*f),[],3),oplt);
add_vectorfield(M,G*f,o,oplt);
M.plt.title = sprintf('Normalized gradient with Vectorfield coloring of the %s function',fname); M.ezvisualize(reshape(normr(G*f),[],3),oplt);
add_vectorfield(M,G*f,o,oplt);
end
function divergence_plot(M,D,oplt,o,vf,vfname)
[vfp,~] = tangent_projection(M,vf,1);
M.plt.title = sprintf('%s vector field & its divergence',vfname); M.ezvisualize(D*vfp(:),oplt);
add_vectorfield(M,vfp,o,oplt);
% oplt.normal_clr = 'b';
% add_vectorfield(M,vf,o,oplt);
add_xyz_axis();
end

