function [M,da] = add_dihedral_ang_visualization(M,exterior,km,r,dis)

% Heuristics to compute the write size displacement and circle radius
N_DIGITS_TO_RND = 2; 
[~,R] = M.box_bounds(); 
if ~exist('exterior','var') ; exterior = 1; end 
if ~exist('r','var') ; r = R/45; end 
if ~exist('dis','var'); dis = R/45; end
if ~exist('km','var'); km = 0; end

% Compute Txt & Arc params: 
M = M.add_face_normals(); [~,~,fe,e_al] = fedge2edge_tables(M); 
[~,da,is_concave] = dihedral_angles_adj(M,1,km); 
if exterior
    da = 360-da;
end

Nfe = size(fe,1); 
assert(Nfe < 800,'Too many face edges - plot will be become incomprehensible'); % TODO: Add a sparisfy option to this function
v1 = M.v(e_al(:,1),:); v2 = M.v(e_al(:,2),:);
ec = (v1+v2)/2; evv = v1 - v2; 
angclr = uniclr('c',Nfe,'dark_pink',find(is_concave)); 

n_at_ec = (M.fn(fe(:,1),:)+M.fn(fe(:,2),:))/2; 
n_at_ec = normr(n_at_ec); 
ectxt = ec + dis*n_at_ec; 
txt = compose('%g^{\\circ}',round(da,N_DIGITS_TO_RND)); 


% Add Text & Arcs: 
hold on; 
text(ectxt(:,1),ectxt(:,2),ectxt(:,3),txt); 
% TODO: Write vector implementation for add_circle 
for i=1:Nfe
    add_circle(ec(i,:),r,angclr(i,:),0,evv(i,:)); 
end
hold off; 
    



end


% for i=1:size(fe,1)
%     Compute center of edge: 
%     v1 = M.v(e(i,1),:); v2 = M.v(e(i,2),:);
%     ec = (v1+v2)/2; 
%     Compute edge vector
%     evv = v1 - v2; 
%     Get relevant dihedral angle
%     ang = da(i); 
%     if ang > 180
%         clr = 'r'; 
%     else 
%         clr = 'c'; 
%     end
%     add_circle(ec,r,clr,0,evv);
%     n_at_ec = (M.fn(fe(i,1),:)+M.fn(fe(i,2),:))/2; 
%     n_at_ec = n_at_ec./normv(n_at_ec); 
%     ectxt = ec + dis*n_at_ec; 
%     text(ectxt(1),ectxt(2),ectxt(3),[num2str(round(ang)),'^{\circ}']); 
% end
% hold off; 
