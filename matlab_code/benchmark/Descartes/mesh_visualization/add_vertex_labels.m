function add_vertex_labels(M,pick,displacement_intensity)

if ~exist('displacement_intensity','var'); displacement_intensity = 0.1; end
if ~exist('pick','var'); pick = []; end

vs = sparsify((1:M.Nv).',pick);
v_labels = cellstr(num2str((vs.').'));
M = M.add_vertex_normals();
dd = displacement_intensity*M.vn(vs,:);
hold on; 
text(M.x(vs)+dd(:,1), M.y(vs)+dd(:,2),M.z(vs)+dd(:,3),v_labels);
hold off;
end
