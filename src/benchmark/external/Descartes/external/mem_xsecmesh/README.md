# xsecmesh.m

Slice a mesh by finding plane/mesh intersection polygons.

<div align="center"><img src="https://cloud.githubusercontent.com/assets/3694352/12079360/6135256e-b1fb-11e5-9bff-b25f6ef01555.png" style="width: 500px;"/></div>

I recently found myself working with some triangulated point-cloud data. 
I wanted to take cross-sections of this mesh, but the available tools weren't 
working. They weren't able to overcome a couple of problems that, it turns out, 
are common in large triangle meshes: >2 faces per edge and 
degenerately-labeled edges. 

In an ideal triangle mesh, any edge (one of the three line sides of a 
triangular face) is incident to exactly two faces. However, point 
triangulation algorithms don't always produce ideal meshes. It's fairly 
common to encounter a mesh that has a few instances in which three faces 
share an edge. It's also not unusual to find degenerate labels in the matrices that 
describe the mesh geometry. That is, they have listed the same vertex or edge 
more than once.

These issues can cause big problems for cross-section algorithms. 
*xsecmesh* overcomes them by first finding edge-plane intersection points and 
then building cross-section polygons using the mesh's connectivity matrix. 
*xsecmesh* can identify multiple, simultaneous polygons that may 
arise in a single cross-section.

*xsecmesh* accepts a plane and a face-vertex mesh as inputs. 

````
plane = [0,0,0.4, 1,0,0, 0,1,0];
polygonCell = xsecmesh(plane, verts, faces);
````

The plane is supplied as a 1x9 
array. Elements 1 through 3 are the Cartesian coordinates of a point on the 
plane. Elements 4 to 6 and 7 to 9 are two vectors that, together with the point, 
define the plane. The mesh must be a face-vertex mesh, meaning that the mesh is 
described by two matrices. One contains all of the vertices in the mesh. The 
other stores the connectivity of these vertices.
