function [new_C]=ProjectCTensor(uf,vf,nf,Old_C,up,vp)
%{ Summary: ProjectCurvatureTensor performs a projection
%of the tensor variables to the vertexcoordinate system
%INPUT:
%uf,vf - face coordinate system
%old_c - face curvature tensor variables as a vector
%up,vp - vertex cordinate system
%OUTPUT:
%new_C - vertex curvature tensor variabels as a column vector
%}
new_CMatrix=zeros(2,2,2);
[r_new_u,r_new_v]=RotateCoordinateSystem(up,vp,nf);

u1=r_new_u*uf;
v1=r_new_u*vf;
u2=r_new_v*uf;
v2=r_new_v*vf;

new_C(1)=Old_C(1)*u1*u1*u1+3*Old_C(2)*u1*u1*v1+3*Old_C(3)*u1*v1*v1+Old_C(4)*v1*v1*v1;
new_C(2)=Old_C(1)*u2*u1*u1+Old_C(2)*(v2*u1*u1+2*u2*u1*v1)+Old_C(3)*(u2*v1*v1+2*u1*v1*v2)+Old_C(4)*v2*v1*v1;
new_C(3)=Old_C(1)*u1*u2*u2+Old_C(2)*(v1*u2*u2+2*u2*u1*v2)+Old_C(3)*(u1*v2*v2+2*u2*v2*v1)+Old_C(4)*v1*v2*v2;
new_C(4)=Old_C(1)*u2*u2*u2+3*Old_C(2)*u2*u2*v2+3*Old_C(3)*u2*v2*v2+Old_C(4)*v2*v2*v2;

end