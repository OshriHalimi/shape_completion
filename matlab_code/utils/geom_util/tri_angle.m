function angle = tri_angle(pt1,pt2,pt3)
    %The function returns the angle opposite to the edge connecting points pt1,pt2 in the
    %triangle created by: pt1,pt2,pt3
    v1 = pt1 - pt3;
    v2 = pt2 - pt3;
    angle = acos(dot(v1,v2)/norm(v1)/norm(v2));
end