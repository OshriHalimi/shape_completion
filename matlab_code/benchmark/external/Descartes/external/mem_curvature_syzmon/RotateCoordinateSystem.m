function [r_new_u,r_new_v]=RotateCoordinateSystem(up,vp,nf)
%{Summary: RotateCoordinateSystem performs the rotation of the vectors up and vp
%to the plane defined by nf
%INPUT:
%up,vp- vectors to be rotated (vertex coordinate system)
%nf - face normal
%OUTPUT:
%r_new_u,r_new_v - rotated coordinate system
%}
r_new_u=up;
r_new_v=vp;
np=cross(up,vp);
np=np/norm(np);
ndot=nf*np';
if ndot<=-1
    r_new_u=-r_new_u;
    r_new_v=-r_new_v;  
    return;
end
perp=nf-ndot*np;
dperp=(np+nf)/(1+ndot);
r_new_u=r_new_u-dperp*(perp*r_new_u');
r_new_v=r_new_v-dperp*(perp*r_new_v');
end