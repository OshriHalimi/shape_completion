% xsecmesh_demo.m
% Load mesh data, get cross-sections, plot the results.

%% MATLAB membrane mesh calculations.
% Mesh obtained from
% blogs.mathworks.com/community/2013/06/20/paul-prints-the-l-shaped-membrane/
load PaulsMembrane.mat;
vertsMembrane = verts;
facesMembrane = faces;
polygonsCellMembrane = cell(1,100);
planeZvals = linspace(0.6, 1.5, 5);
for numPlane = 1:numel(planeZvals)
    % Create a plane.
    planePoints = [
        0, 0, 0 + planeZvals(numPlane), ...
        1, 0, 0 + planeZvals(numPlane), ...
        0, 1, 0 + planeZvals(numPlane)
        ];
    plane = [
        planePoints(1:3), ...
        planePoints(4:6)-planePoints(1:3), ...
        planePoints(7:9)-planePoints(1:3)
        ];
    % Get the plane/mesh cross-section.
    polyCellNow = xsecmesh(plane, vertsMembrane, facesMembrane);
    % Store the results.
    if ~isempty(polyCellNow)
        % Find the first un-used cell array element.
        firstIx = find(cellfun(@isempty,polygonsCellMembrane),1);
        % Store the polygons.
        polygonsCellMembrane(firstIx:firstIx+numel(polyCellNow)-1) = polyCellNow;
    end
end
% Remove the empty cells.
polygonsCellMembrane = polygonsCellMembrane(~cellfun(@isempty,polygonsCellMembrane));

%% Statue mesh calculations.
% This mesh data comes from the SketchUp model at
% 3dwarehouse.sketchup.com/model.html?id=c7f8c1e659f73b68cf004563556ddb36
% which was exported in .off format before importing in MATLAB.
load('statue.mat');
vertsStatue = verts;
facesStatue = faces;
polygonsCellStatue = cell(1,25);
% Create a bunch of planes. All are normal to z-hat to make things simple.
planeZvals = linspace(0,0.4,4);
for numPlane = 1:numel(planeZvals)
    % Create a plane.
    planePoints = [
        0,0,0+planeZvals(numPlane), ...
        1,0,0+planeZvals(numPlane), ...
        0,1,0+planeZvals(numPlane)
        ];
    plane = [
        planePoints(1:3), ...
        planePoints(4:6)-planePoints(1:3), ...
        planePoints(7:9)-planePoints(1:3)
        ];
    % Get the plane/mesh cross-section.
    polyCellNow = xsecmesh(plane, vertsStatue, facesStatue);
    % Store the results.
    if ~isempty(polyCellNow)
        % Find the first un-used cell array element.
        firstIx = find(cellfun(@isempty,polygonsCellStatue),1);
        % Store the polygons.
        polygonsCellStatue(firstIx:firstIx+numel(polyCellNow)-1) = polyCellNow;
    end
end
% Remove the empty cells.
polygonsCellStatue = polygonsCellStatue(~cellfun(@isempty,polygonsCellStatue));


fullfig;
    clf;
    hs1 = subplot(2,2,1);
        hMembrane = patch('vertices', vertsMembrane, 'faces', facesMembrane);
        set(                                                ...
            hMembrane           ,                           ...
            'FaceColor'         ,   [0.8,0.8,1]         ,   ...
            'EdgeColor'         ,   1/255*[59,59,59]    ,   ...
            'FaceLighting'      ,   'gouraud'           ,   ...
            'AmbientStrength'   ,   0.15                ,   ...
            'LineWidth'         ,   1                   ,   ...
            'FaceAlpha'         ,   0.7                     ...
            );
        grid on;
        set(gca,'XTickLabel',[],'YTickLabel',[],'ZTickLabel',[]);
        material('dull');
        camlight('headlight');
        axis equal;
    % Plot the cross-sections.
    hs2 = subplot(2,2,2);
        for numPoly = 1:numel(polygonsCellMembrane)
            color = 'none';
            hold on
            hPoly = patch(                  ...
                polygonsCellMembrane{numPoly}(:,1), ...
                polygonsCellMembrane{numPoly}(:,2), ...
                polygonsCellMembrane{numPoly}(:,3), 'k');
            set(                            ...
                hPoly                   ,   ...
                'FaceColor' ,   color   ,   ...
                'EdgeColor' ,   'k'     ,   ...
                'LineWidth' ,   2       ,   ...
                'FaceAlpha' ,   0.7         ...
                );
        end
        grid on;
        set(gca,'XTickLabel',[],'YTickLabel',[],'ZTickLabel',[]);
        axis equal;
    hs3 = subplot(2,2,3);
        hStatue = patch('vertices', vertsStatue, 'faces', facesStatue);
        set(                                                ...
            hStatue             ,                           ...
            'FaceColor'         ,   143/255*[1,1,1]     ,   ...
            'EdgeColor'         ,   'none'              ,   ...
            'FaceLighting'      ,   'gouraud'           ,   ...
            'AmbientStrength'   ,   0.15                    ...
            );
        material('dull');
        camlight('headlight');
        lightangle(10,-90);
        grid on;
        set(gca,'XTickLabel',[],'YTickLabel',[],'ZTickLabel',[]);
        axis equal;
    hs4 = subplot(2,2,4);
        for numPoly = 1:numel(polygonsCellStatue)
            color = 'none';
            hold on
            hPoly = patch(                  ...
                polygonsCellStatue{numPoly}(:,1), ...
                polygonsCellStatue{numPoly}(:,2), ...
                polygonsCellStatue{numPoly}(:,3), 'k');
            set(                            ...
                hPoly                   ,   ...
                'FaceColor' ,   color   ,   ...
                'EdgeColor' ,   'k'     ,   ...
                'LineWidth' ,   2           ...
                );
        end
        grid on;
        set(gca,'XTickLabel',[],'YTickLabel',[],'ZTickLabel',[]);
        axis equal;
    set([hs1,hs2],'View',[-105,15]);
    set(hs2,'XLim',[-0.2,1.2],'YLim',[-0.2,1.2],'ZLim',[0.5,1.8]);
    set([hs3,hs4],'View',[16,11]);
    set([hs3,hs4],'XLim',[-0.2,0.2],'YLim',[-0.2,0.2],'ZLim',[0,0.5]);
    set(gcf,'Color','w');