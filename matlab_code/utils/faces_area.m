function area = faces_area(vertices,faces)
%INPUT: vertices - #vertices X 3 matrix of vertices coordinates: X,Y,Z
%       faces    - #faces X 3 matrix of composing vertices indices for every face
%OUTPUT: area - per face area vector
cross_product_mat = cross(vertices(faces(:,2),:) - vertices(faces(:,1),:),vertices(faces(:,3),:) - vertices(faces(:,2),:),2);
area = vecnorm(cross_product_mat,2,2);
end

