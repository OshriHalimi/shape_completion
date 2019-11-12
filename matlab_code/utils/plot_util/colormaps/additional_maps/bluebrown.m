function [cmap] = bluebrown(m)
if ~exist('m','var'); m = 512; end
cmap = create_clr_map(m,[71,9,15;255,255,255;73,97,187],[0.7],0); 
end
