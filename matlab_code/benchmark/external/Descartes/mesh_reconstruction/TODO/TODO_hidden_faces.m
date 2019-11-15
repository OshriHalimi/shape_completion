% M = M.add_face_normals();
% M = M.add_face_areas();
% M = M.add_vertex_normals();id = [];
% [~,v] = M.clean_manifold();
% for i=1:numel(v)
%     curr_v = v(i); 
%     prod = abs(dot(M.fn,repmat(M.vn(curr_v,:),M.Nf,1),2)); 
%     id = [id ;find(prod>0.999)]; 
% end
% fia = M.fa(fi); fi = fi(fia<314);

[DA,~] = M.dihedral_angles_adj();
[rr,cc,vals]=find(DA<=185 & DA >=175); 
fi = unique([rr ; cc;]);
M = remove_faces(M,fi);