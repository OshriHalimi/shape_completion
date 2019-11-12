function P = add_edge_visualization(M,E,is_fedge,clr)

if ~exist('is_fedge','var'); is_fedge = 0; end
if ~exist('clr','var') || isempty(clr); clr = 'c'; end

E = E.';
hold on;
if is_fedge
   M = M.add_face_centers(); 
    x = M.fc(:,1); y = M.fc(:,2); z = M.fc(:,3); 
%      M = M.add_face_normals();
%     x = M.fc(:,1)+0.01*M.fn(:,1); y = M.fc(:,2)+0.01*M.fn(:,2); z = M.fc(:,3)+0.01*M.fn(:,3);
else
   x = M.v(:,1); y = M.v(:,2); z = M.v(:,3);
end
P = plot3(x(E), y(E), z(E), 'Color',clr, 'LineWidth',1.5);
hold off;
end