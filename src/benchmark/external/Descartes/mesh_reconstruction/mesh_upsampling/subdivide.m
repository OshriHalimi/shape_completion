function [M_new] = subdivide(M,n)
% TODO - create demo for this 
% Vectorized Triangle Subdivision: split each triangle 
% input face into four new triangles. 
if ~exist('n','var'); n = 1; end
v = M.v;
f = M.f; 

for i=1:n
% disp(['Input mesh: ' num2str(numfaces) ' triangles, ' ... 
%     num2str(numverts) ' vertices.']);
fk1 = f(:,1);
fk2 = f(:,2);
fk3 = f(:,3);
% create averages of pairs of vertices (k1,k2), (k2,k3), (k3,k1)
    m1x = (v( fk1,1) + v( fk2,1) )/2;
    m1y = (v( fk1,2) + v( fk2,2) )/2;
    m1z = (v( fk1,3) + v( fk2,3) )/2;
    
    m2x = (v( fk2,1) + v( fk3,1) )/2;
    m2y = (v( fk2,2) + v( fk3,2) )/2;
    m2z = (v( fk2,3) + v( fk3,3) )/2;
    
    m3x = (v( fk3,1) + v( fk1,1) )/2;
    m3y = (v( fk3,2) + v( fk1,2) )/2;
    m3z = (v( fk3,3) + v( fk1,3) )/2;
    
vnew = [ [m1x m1y m1z]; [m2x m2y m2z]; [m3x m3y m3z] ];
clear m1x m1y m1z m2x m2y m2z m3x m3y m3z
[vnew_,~,jj] = unique(vnew, 'rows' );
clear vnew; 
m1 = jj(1:M.Nf)+M.Nv;
m2 = jj(M.Nf+1:2*M.Nf)+M.Nv;
m3 = jj(2*M.Nf+1:3*M.Nf)+M.Nv;
tri1 = [fk1 m1 m3];
tri2 = [fk2 m2 m1];
tri3 = [ m1 m2 m3];
tri4 = [m2 fk3 m3];
clear m1 m2 m3 fk1 fk2 fk3
 
v1 = [v; vnew_]; % the new vertices
f1 = [tri1; tri2; tri3; tri4]; % the new faces
f = f1; 
v = v1;
M.Nf = size(f1,1); 
M.Nv = size(v1,1); 
end
disp(['Output mesh: ' num2str(size(f1,1)) ' triangles, ' ... 
    num2str(size(v1,1))  ' vertices.']);
M_new = Mesh(v1,f1,M.name,M.path); 
