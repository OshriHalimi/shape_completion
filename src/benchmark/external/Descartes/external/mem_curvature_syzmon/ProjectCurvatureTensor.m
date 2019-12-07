function [new_ku,new_kuv,new_kv]=ProjectCurvatureTensor(uf,vf,nf,old_ku,old_kuv,old_kv,up,vp)
%{ Summary: ProjectCurvatureTensor performs a projection
%of the tensor variables to the vertexcoordinate system
%INPUT:
%uf,vf - face coordinate system
%old_ku,old_kuv,old_kv - face curvature tensor variables
%up,vp - vertex cordinate system
%OUTPUT:
%new_ku,new_kuv,new_kv - vertex curvature tensor variabels
%}
[r_new_u,r_new_v]=RotateCoordinateSystem(up,vp,nf);
OldTensor=[old_ku old_kuv; old_kuv old_kv];
u1=r_new_u*uf;
v1=r_new_u*vf;
u2=r_new_v*uf;
v2=r_new_v*vf;
new_ku=[u1 v1]*OldTensor*[u1;v1];
new_kuv=[u1 v1]*OldTensor*[u2;v2];
new_kv=[u2 v2]*OldTensor*[u2;v2];
%{
new_ku=old_ku*u1*u1+2*old_kuv*(u1*v1)+old_kv*v1*v1;
new_kuv=old_ku*u1*u2+old_kuv*(u1*v2+u2*v1)+old_kv*v1*v2;
new_kv=old_ku*u2*u2+2*old_kuv*(u2*v2)+old_kv*v2*v2;
%}
end
