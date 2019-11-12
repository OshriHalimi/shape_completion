function add_feature_pts(M,pts,scale_fact,clr)

% Set defaults 
if ~exist('scale_fact','var') || isempty(scale_fact); scale_fact = 0.05; end
if ~exist('clr','var'); clr = 'k'; end

% Compute how much to scale the radius by
sphere_vol = 4*pi/3;
V = M.volume(); 
scale = scale_fact*(V/sphere_vol)^(1/3);

% Create and scale the sphere
[Xs,Ys,Zs] = sphere(40); 
Xs = scale*Xs; Ys = scale*Ys; Zs = scale*Zs; 

% Plot them 
hold on;
for i=1:size(pts,1)
    surf(Xs+pts(i,1),Ys+pts(i,2),Zs+pts(i,3),'EdgeColor','none','FaceColor',clr);
end
hold off; 
axis auto;