function r = autocrop_image(fname, delete_source)

if nargin==1
    delete_source = false;
end

img = imread(fname);

whites = img(:,:,1) == 255 & img(:,:,2) == 255 & img(:,:,3) == 255;
[i,j] = find(~whites);

top = min(i);
left = min(j);
bottom = max(i);
right = max(j);

r = [left-1, top-1, right-left+2, bottom-top+2];

%figure, imshow(img), hold on, rectangle('Position',r)

cropped = imcrop(img,r);

%figure, imshow(cropped);

[d,n,e] = fileparts(fname);
out_fname = sprintf('%s/%s_crop%s',d,n,e);

imwrite(cropped, out_fname);

if delete_source
    delete(fname);
    copyfile(out_fname, fname, 'f');
    delete(out_fname);
end

end
