function [cmap] = bluewhite(m)
if ~exist('m','var'); m = 512; end
cmap = create_clr_map(m,[255,255,255;255,255,255; 77,191,237],[0.5],0); 
end

