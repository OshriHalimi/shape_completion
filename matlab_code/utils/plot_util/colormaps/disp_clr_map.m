function disp_clr_map(cmap)

if ischar(cmap) 
    cmap = eval(cmap); 
end
if isnumeric(cmap)
    cmap = reshape(cmap,[],1,3); 
    fullfig; imshow(repmat(cmap,1,100,1)); 
else
    error('Invalid Cmap'); 
end
end

