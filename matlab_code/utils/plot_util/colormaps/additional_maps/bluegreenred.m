function [cmap] = bluegreenred(m)
if ~exist('m','var'); m = 512; end
cmap = create_clr_map(m,[0,0,1;0,1,0; 1,0,0],[0.5],0); 
end

