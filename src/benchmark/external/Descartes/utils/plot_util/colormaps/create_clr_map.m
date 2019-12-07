function cmap = create_clr_map(N,clrs,ticks,to_disp)
% N - Number of points
% stop_points - The color changes, from left to right
% ticks - Fractions from 0 -> 1 - When to add intermediate stop points

if ~exist('ticks','var'); ticks = []; end
if ~exist('to_disp','var'); to_disp = 0; end

N_clrs = size(clrs,1); 
n_over_1 = any(clrs(:)>1); 
if n_over_1 
    clrs = clrs./255; 
end
N_ticks = length(ticks); 
assert(N_clrs == N_ticks +2); 

ticks_in_pts = round(N*ticks); 
[X,Y] = meshgrid(1:3,1:N); 
cmap = interp2(X([1,ticks_in_pts,N],:),Y([1,ticks_in_pts,N],:),clrs,X,Y); %// interpolate colormap

if to_disp
    disp_clr_map(cmap);
end
