% Example 1

grid=1:StepSize:GridSize;
[x,y]=meshgrid(grid,grid);
FV.faces = delaunay(x,y);
z = peaks(size(grid,2));
FV.vertices=[x(1:end)',y(1:end)',z(1:end)'];

% Example 2: 
[x,y,z] = peaks(n); % Built-in Matlab function 
[f,v] = surf2patch(x,y,z,'triangles');

% Example 3: 
[~, v] = bucky;
f = convhull(v);