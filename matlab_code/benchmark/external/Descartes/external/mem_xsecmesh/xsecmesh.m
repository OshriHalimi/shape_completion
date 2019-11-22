function polyCell = xsecmesh(plane, verts, faces, varargin)
% XSECMESH Find polygon(s) formed by a cross-section between a mesh and a plane.
% 
% XSECMESH(plane, verts, faces, nSigFig) Generate closed polygon(s) 
% representing the cross-section resulting from the intersection of a 
% triangle mesh and a plane.
% 
% Inputs:
%   plane       A plane given in the form [x0,y0,z0,vx1,vy1,vz1,vx2,vy2,vz2]. 
%               The first 3 elements are a point on the plane. Elements 4:6 and 
%               7:9 each represent two different in-plane vectors.
%   verts       Vertices matrix for the mesh.
%   faces       Faces matrix for the mesh.
% Optional input:
%   nSigFig     Truncate coordinate values to nSigFig digits.
% 
% Output:
%   polyCell    Cell array of closed cross-section polygons.
% 
% Note: Cross-section is not calculated when an edge end-point (a vertex of 
% the solid) lies on the intersection plane.
%
% License: MIT
% Brian Hannan 
% brianmhannan@gmail.com
% Written while working under the direction of Dr. Doug Rickman at the NASA 
% Marshall Space Flight Center.
% Tested on MATLAB R2012a and 2015b (OS-X and Windows).
% Latest edit 31 Dec. 2015.

% Handle the optional input argument, nSigFig.
numVarArgs = length(varargin);
if numVarArgs > 1
    error('myfuns:xsecmesh:TooManyInputs', ...
        'This function takes at most 1 optional input.');
end
optArgs = {6};
optArgs(1:numVarArgs) = varargin;
nSigFig = optArgs{:};

% Require triangular mesh input.
NUM_EDGES_PER_FACE = 3;
% edgeCheckedMat is a list that will be populated with the end-poinds of 
% edges that have been checked for intersection.
edgeCheckedMat = zeros(3*size(faces,1),6);
edgeCheckedIx = 1;
vertsCell = mat2cell(verts, ones(1, size(verts, 1)), size(verts, 2));
distArray = cellfun(@calc_plane_point_distance, vertsCell, ...
                                        repmat({plane},size(vertsCell)));
if any(truncate_matrix_values(~distArray, nSigFig))
    polyCell = {}; % Return empty cell.
    disp(['Mesh cross-section operation terminated because a mesh vertex '...
        'lies on the slicing plane.']);
else
    % Each row of intPtsAndNbrFaces will contain the [x,y,z] coordinates of 
    % an intersection point in 1:3. In the same row, elements 4:5 hold the 
    % ixs of the two faces that the point point lies on.
    % Why combine intersection point coords and face labels in one matrix?  
    % Repeated entries will need to be removed. This operation is simpler
    % when the data are stored in a single matrix.
    intPtsAndNbrFaces = zeros(NUM_EDGES_PER_FACE*size(faces,1), 5);
    usedRows = zeros(1, size(intPtsAndNbrFaces,2));
    for faceNum = 1:size(faces, 1)
        tfFaceIntersect = test_face_plane_intersect(verts,faces,faceNum,plane);
        if tfFaceIntersect % Find the intersection points' coordinates.
            vertIxsCurrentFace = faces(faceNum, :);
            faceVertIxList = [vertIxsCurrentFace, vertIxsCurrentFace(1)];
            for edgeNum = 1:NUM_EDGES_PER_FACE
                p1 = verts(faceVertIxList(edgeNum), :);
                p2 = verts(faceVertIxList(edgeNum+1), :);
                tfEdgeChecked = query_edge_intersect_checked(p1, p2, edgeCheckedMat);
                if ~tfEdgeChecked
                    % Add this edge to the list of edges tested for intersect.
                    edgeCheckedMat(edgeCheckedIx, :) = [p1, p2];
                    edgeCheckedIx = edgeCheckedIx + 1;
                    tfEdgeIntersect = test_edge_plane_intersect(p1, p2, plane);
                    if tfEdgeIntersect
                        usedRows(4*(faceNum-1)+edgeNum) = 1;
                        intPtNow = get_line_plane_intersect(p1, p2, plane);
                        ixsFacesContainingPoint = find_face_ixs_from_edge(p1,...
                            p2, verts, faces);
                        assert(numel(ixsFacesContainingPoint) == 2, ...
                            sprintf(['Current intersection point lies on ' ...
                                '%d faces. Expected result is 2.'], ...
                                numel(ixsFacesContainingPoint)));
                        intPtsAndNbrFaces(4*(faceNum-1)+edgeNum, :) = ...
                            [intPtNow(1), intPtNow(2), intPtNow(3), ...
                            ixsFacesContainingPoint];
                    end
                end
            end
        end
    end % faceNum
    % Truncate coord vals.
    intPtsAndNbrFaces = truncate_matrix_values(intPtsAndNbrFaces,nSigFig);
    intPtsAndNbrFaces = unique(intPtsAndNbrFaces(logical(usedRows)',:), 'rows');
    % Separate intPts, nbrFaces.
    nbrFaces = intPtsAndNbrFaces(:, 4:5);
    intPts = intPtsAndNbrFaces(:, 1:3);
    % Check that no intersection points coincide with a vertex after values
    % are truncated. In order to determine vertex, intPoint equality, the same 
    % truncation operation is performed on both.
    tfIntPointVertexEqual = any(ismember(...
        truncate_matrix_values(verts,nSigFig), intPts, 'rows'));
    if ~tfIntPointVertexEqual
        % Pass intersection points matrix and connectivity matrix nbrFaces 
        % to buildSectionPolys to generate a cell of polygons.
        polyCell =  build_cross_sec_polygons(intPts,nbrFaces);
    else
        polyCell = {};
    end
end
end % main


function polyCell = build_cross_sec_polygons(intPts, nbrFaces)
% BUILD_CROSS_SEC_POLYGONS constructs closed polygon(s) for a plane of section
% from a list of intersection points and their connectivity.
% 
% build_cross_sec_polygons(intPts, nbrFaces) returns a cell of polygons.
% Polygons are generated from the vertices in intPts and their
% connectivity, contained in nbrFaces.
% 
% Inputs:
%   intPts      Nx3 matrix of cartesian points.
%   nbrFaces    Nx2 matrix. Row k contains the ixs of the two faces
%               that neighbor the point intPts(k,:).
% 
% Output:
%   polyCell    Cell array of polygons. Each cell in polyCell holds 
%               one polygon.

% Once the points of intersection are found for one "slice", these points 
% must be connected to draw the polygon(s) that represent cross-
% section(s). (Note: an "intersection point" is found by calculating the 
% intersection between an edge and the cutting plane.) Each face of the
% triangular mesh that intersects the plane must produce two intersection
% points (faces that intersect at one point or are coplanar with the
% cutting plane are discarded). Therefore, polygons may be constructed by
% joining intersection points that lie on the same face.

% Preallocate cell for multiple polygon output. Max no. polygons is 1/3 
% number of intersection points.
polyCell = cell(1, floor(size(intPts,1)/3));
isPtCheckedArray = false(size(intPts,1), 1);
newPoly = true;
nPoly = 0;
loopCount = 0;
while ~all(isPtCheckedArray)
    if newPoly
        intPtCount = 1;
        % Preallocate polyNowPts so it may hold all remaining int pts.
        polyNowPts = nan(sum(~isPtCheckedArray), 3);
        % Get ix of the first unchecked point. Store this poly's 1st pt.
        ixIntPointNow = find(~isPtCheckedArray, 1);
        polyNowFirstPtIx = ixIntPointNow;
        % Temporarily mark start point as checked.
        isPtCheckedArray(polyNowFirstPtIx) = true;
        % Count nPoly, the polygon count for this plane of section.
        nPoly = nPoly + 1;
        % An int pt belongs to 2 faces. We can select either as the 
        % "current face" since this choice is equivalent to choosing to 
        % traverse CW or CCW about the polygon. Pick element 1.
        ixFaceNow = nbrFaces(ixIntPointNow, 1);
        newPoly = false;
    end
    % At 3rd step, set start point to unchecked so the polygon can close.
    % Start point labeled as "checked" at intPtCount=3 to prevent back-tracking.
    if intPtCount == 3
        isPtCheckedArray(polyNowFirstPtIx) = false;
    end 
    % Get coords of this intersection point. Store in polyNowPts.
    intPtNow = intPts(ixIntPointNow, :);
    polyNowPts(intPtCount, :) = intPtNow;
    isPtCheckedArray(ixIntPointNow) = true;
    % Identify ixs of rows in nbrFaces that contain current face ix, ixFaceNow.
    rowIxFaceNowArray = find(sum(nbrFaces==ixFaceNow, 2));
    assert(numel(rowIxFaceNowArray)==2, ...
        sprintf(['While creating the intersection polygon, an intersection '...
        'point was found to lie on %d faces. Expected result is 2.'], ...
        numel(rowIxFaceNowArray)));
    % Find the other row in nbrFaces that also contains ixFaceNow.
    ixIntPointNext = rowIxFaceNowArray(rowIxFaceNowArray ~= ixIntPointNow);
    ixsFacesNeighboringNextPoint = nbrFaces(ixIntPointNext, :);
    % Ix of next face, ixFaceNext, is the element in 
    % ixsFacesNeighboringNextPoint that is not equal to ixFaceNow.
    ixFaceNext = ixsFacesNeighboringNextPoint(ixsFacesNeighboringNextPoint ~=...
        ixFaceNow);
    ixFaceNow = ixFaceNext; % On to the next intersection point.
    ixIntPointNow = ixIntPointNext;
    intPtCount = intPtCount + 1;
    if ixIntPointNow == polyNowFirstPtIx
        % Polygon is complete. Close, store it in output cell.
        polyNowPts(intPtCount,:) = intPts(polyNowFirstPtIx,:);
        % Remove any un-used, preallocated rows.
        polyNowPts = polyNowPts(1:intPtCount,:);
        % A polygon must have at least 3 unique vertices. If less than 3
        % are identified, do not output a polygon. < 3 vertices may be
        % present for very small polygons that have been reduced to 1 or 2 
        % points after values are truncated.
        if size(unique(polyNowPts,'rows') > 2)
            polyCell{nPoly} = polyNowPts;
        end
        % Mark start as "checked". All pts on polygon have now been checked.
        isPtCheckedArray(ixIntPointNow) = true;
        newPoly = true;
    end
    if loopCount > size(intPts,1)
        error('myfuns:xsecmesh:loopRunaway', ...
            'Failed to ID next point in polygon.');
    end
    loopCount = loopCount + 1;
end
% Remove empty cells.
polyCell = polyCell(~cellfun(@isempty,polyCell));
end


function dist = calc_plane_point_distance(point, plane)
% Calculate the shortest distance between a point [x,y,z] and a plane 
% [xp,yp,zp,xv1,yv1,zv1,xv2,yv2,zv2].
% Distance is found by calculating the projection of w (a vector from a
% point on the plane to the query point) onto the plane's normal vector.
% Returns signed distance.
planeNormUV = cross(plane(4:6), plane(7:9)) ./ ...
    norm(cross(plane(4:6),plane(7:9)));
w = -[plane(1)-point(1), plane(2)-point(2), plane(3)-point(3)];
dist = dot(planeNormUV,w);
end


function faceIxs = find_face_ixs_from_edge(point1, point2, vertsMat, facesMat)
% Find the ixs of all mesh faces in facesMat that have an edge defined by the 
% points point1 and point2.
p1VertsRowIx = find_row_in_matrix(point1, vertsMat);
p2VertsRowIx = find_row_in_matrix(point2, vertsMat);
% Get matrices with dims equal to facesMat. If anentry in this mat equals 1, 
% then one of these points is located here.
tfP1InFacesMat = ismember(facesMat, p1VertsRowIx);
tfP2InFacesMat = ismember(facesMat, p2VertsRowIx);
tfP1P2InFacesMat = tfP1InFacesMat + tfP2InFacesMat;
faceIxs = find(sum(tfP1P2InFacesMat,2) == 2)';
end


function tfIntersect = test_face_plane_intersect(vertsMat,facesMat,faceIx,myPlane)
pointPlaneDistanceMat = [
    calc_plane_point_distance(vertsMat(facesMat(faceIx,1),:), myPlane), ...
    calc_plane_point_distance(vertsMat(facesMat(faceIx,2),:), myPlane), ...
    calc_plane_point_distance(vertsMat(facesMat(faceIx,3),:), myPlane)
    ];
pointPlaneDistanceMat = truncate_matrix_values(pointPlaneDistanceMat, 13);
vertDists_isPos = pointPlaneDistanceMat > 0;
vertDists_isNeg = pointPlaneDistanceMat < 0;
tfIntersect = sum(vertDists_isPos)>0 && sum(vertDists_isNeg)>0;
end


function tfIntersect = test_edge_plane_intersect(p1, p2, myPlane)
% Use signed endpt-plane dist to identify edge/plane intersect.
% Return true if the line segment with endpoints p1 and p2 intersects the 
% plane myPlane. Returns false if the segment does not intersect or if the 
% line segment lies on the plane. myPlane has the form 
% [x0,y0,z0,vx1,vy1,vz1,vx2,vy2,vz2].
edgeEndPlaneDists = [calc_plane_point_distance(p1, myPlane), ...
    calc_plane_point_distance(p2, myPlane)];
isPosEdgeEndDists = edgeEndPlaneDists > 0;
isNegEdgeEndDists = edgeEndPlaneDists < 0;
tfIntersect = sum(isPosEdgeEndDists)>0 && sum(isNegEdgeEndDists)>0;
end


function intersectPoint = get_line_plane_intersect(linePt1, linePt2, myPlane)
% Find the intersection between a line containing the points linePt1 & linePt2 
% and the plane myPlane. See mathworld.wolfram.com/Line-PlaneIntersection.html.
p1 = myPlane(1:3);
p2 = p1 + myPlane(4:6);
p3 = p1 + myPlane(7:9);
A = [
    1, 1, 1, 1;
    p1(1), p2(1), p3(1), linePt1(1);
    p1(2), p2(2), p3(2), linePt1(2);
    p1(3), p2(3), p3(3), linePt1(3);
    ];
B = [
    1, 1, 1, 0;
    p1(1), p2(1), p3(1), linePt2(1)-linePt1(1);
    p1(2), p2(2), p3(2), linePt2(2)-linePt1(2);
    p1(3), p2(3), p3(3), linePt2(3)-linePt1(3);
    ];
t = -det(A)/det(B);
intersectPoint = [
    linePt1(1) + (linePt2(1) - linePt1(1))*t,
    linePt1(2) + (linePt2(2) - linePt1(2))*t,
    linePt1(3) + (linePt2(3) - linePt1(3))*t
    ];
end


function tfChecked = query_edge_intersect_checked(p1, p2, checkedEdgesMat)
% Edges generally belong to more than one face. When looking for face/plane 
% intersection, avoid repeated operations by comparing the current edge to a 
% list of previously checked edges.
% The inputs p1 and p2 are 1x3 vectors representing cartesian points. 
% checkedEdgesMat is a matrix of size Nx6. Each row is formed by horizontally 
% concatenating two points. Look for equivalent rows of the form [p1,p2] and 
% [p2,p1].
is_row_in_matrix = @(myRow) any(ismember(checkedEdgesMat, ...
    repmat(myRow, size(checkedEdgesMat,1), 1), 'rows'));
tfChecked = is_row_in_matrix([p1, p2]) || is_row_in_matrix([p2, p1]);
end


function rowIx = find_row_in_matrix(myRow, myMatrix)
rowIx = find(ismember(myMatrix, repmat(myRow,size(myMatrix,1),1), 'rows'));
assert(~isempty(rowIx), 'Searched for a matrix row that does not exist.');
end


function roundedMat = truncate_matrix_values(myMatrix, nSigFig)
% Truncate all numerical values in a matrix. Rounds all elements of 
% myMatrix to nSigFig significant digits.
roundedMat = arrayfun(@(val,nsf) round(val*10^(nsf-1))/10^(nsf-1), ...
    myMatrix, nSigFig.*ones(size(myMatrix)));
end