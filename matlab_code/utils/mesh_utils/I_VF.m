function IVF = I_VF(faces,vertices)
    % The function returns the interpolation matrix from faces to vertices
    farea = faces_area(vertices,faces);
    varea = vertex_area(vertices,faces);
    IVF = (1/3)*(1./varea)*farea';
    IVF = adjacency_VF(vertices,faces).*IVF;
end

