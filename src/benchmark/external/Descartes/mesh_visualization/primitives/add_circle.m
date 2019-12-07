function h = add_circle(c,r,clr,is_filled,pnormal)

if ~exist('clr','var'); clr = 'teal'; end
if ~exist('r','var'); r = 1; end
if ~exist('is_filled','var'); is_filled = 0; end
clr = uniclr(clr); 

[~,el] = view;
if el == 90 && ~exist('pnormal','var')
    if ~exist('c','var'); c=[0,0]; end
    h=add_circle_2D(c,r,clr,is_filled);
else
    if ~exist('pnormal','var'); pnormal = [1,0,0]; end
    if ~exist('c','var'); c=[0,0,0]; end
    h=add_circle_3D(c,r,clr,pnormal,is_filled);
end
end
%-------------------------------------------------------------------------%
%
%-------------------------------------------------------------------------%
function h =add_circle_2D(c,r,clr,is_filled)
theta=linspace(0,2*pi,100); rho=ones(1,100)*r;
[X,Y] = pol2cart(theta,rho);
X=X+c(1); Y=Y+c(2);
hold on;
if is_filled
    h=fill(X,Y,clr);
else
    h = plot(X,Y,'Color',clr);
end
hold off;
end
function h =add_circle_3D(c,r,clr,pnormal,is_filled)
theta=linspace(0,2*pi,100);
v=null(pnormal); % All (2) vectors in 3D orthogonal to pnormal
p=repmat(c',1,size(theta,2))+r*(v(:,1)*cos(theta)+v(:,2)*sin(theta));
hold on;
if is_filled
    h = fill3(p(1,:),p(2,:),p(3,:),clr);
else
    h = plot3(p(1,:),p(2,:),p(3,:),'Color',clr);
end
hold off;

end

