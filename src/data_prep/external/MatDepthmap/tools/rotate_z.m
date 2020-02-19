function R = rotate_z(a)
a = -pi*a/180;
R = [cos(a) -sin(a) 0 ; sin(a) cos(a) 0 ; 0 0 1];
end