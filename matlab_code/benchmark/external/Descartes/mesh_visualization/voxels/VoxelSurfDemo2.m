clear;
%This script creates a random voxel landscape out of hills and lakes that is
%covered with some trees. The trees are simply modelled as spheres plus
%cylindrical trunks.

rng(5);

%init parameters:
%lanscape size:
XX=256;
YY=256;
ZZ=128;

TreeHeight=24;
TreeWidth=16;
trunkDiameter=5;
grassHeight=5;
waterLevel=51;
MeanTerrainHeight=55;
heightmapSmoothing=35;
HeightVariationAmplifier=20;
nTreeAttempts=50;

%RGB colors:
grasscolor=[0.1   0.6 0.2];
stonecolor=[0.6 0.6 0.6];
watercolor=[0.3   0.3    0.9];
trunkcolor=[0.3 0.2 0.1];
leavescolor=[ 0 0.3 0];

%create empty data arrays; water as seperate transparent layer:
voxels=zeros(YY,XX,ZZ);
watervoxels=zeros(YY,XX,ZZ);
[Xcoords,Ycoords,Zcoords]=meshgrid(1:XX,1:YY,1:ZZ);
[Xc,Yc]=meshgrid(1:XX,1:YY);

%use a smoothened random noise map as heightmap:
heightmap=ZZ*rand(YY,XX);
heightmap=imgaussfilt(heightmap,heightmapSmoothing);
%amplify height variations and smooth them at the corners:
heightmap=MeanTerrainHeight+HeightVariationAmplifier*(0.6-((Xc/XX-1/2).^2+(Yc/YY-1/2).^2)).*(heightmap-ZZ/2);


%Set stone and grass voxels:
voxels(Zcoords<repmat(heightmap,1,1,ZZ)) =2;
voxels(Zcoords<repmat(heightmap,1,1,ZZ)-grassHeight) =1;
watervoxels((voxels == 0) &( Zcoords<waterLevel))=3;

%create trees out of cylinder trunks and sphere leaves:
[treeX,treeY,treeZ]=meshgrid(1:TreeWidth,1:TreeWidth,1:TreeHeight);
trunk=((treeX-TreeWidth/2).^2 +(treeY-TreeWidth/2).^2 < (trunkDiameter/2)^2).*(treeZ<TreeHeight*0.6)*4;
leaves = ((treeX-TreeWidth/2).^2 +(treeY-TreeWidth/2).^2 + (treeZ-TreeHeight*0.6).^2 < (TreeWidth/2)^2)*5;
tree=max(trunk,leaves);

%random tree positions excluding the edge of the map:
TreePos = 1+TreeWidth/2+ floor(rand(2,nTreeAttempts).*[XX-TreeWidth;YY-TreeWidth]);

for t=1:nTreeAttempts
    
    x=TreePos(1,t);
    y=TreePos(2,t);
    z= round(heightmap(y,x))-1;
    
    if z> waterLevel-1 %test if on land
       v=voxels(y-TreeWidth/2+1:y+TreeWidth/2,...
              x-TreeWidth/2+1:x+TreeWidth/2,...
              z:z+TreeHeight-1);
          
       if sum(v(:)>3) == 0 %test if there is no other tree
           voxels(y-TreeWidth/2+1:y+TreeWidth/2,...
                  x-TreeWidth/2+1:x+TreeWidth/2,...
                  z:z+TreeHeight-1)=max(v,tree);
       end
    end
    
end

figure(1);
clf;
%plot all solid voxels
voxelSurf(voxels,true);
%plot water voxels with transparency:
hold on
voxelSurf(watervoxels,false,[1 XX 1 YY 1 ZZ],0.5);
hold off
axis([0 XX+1 0 YY+1 0 ZZ+1]);
camproj('perspective')
%set the colors defined above:
colormap([stonecolor;grasscolor;watercolor;trunkcolor;leavescolor]);
%add sunlight:
light('Position',[128 0 256],'Style','local')
%%
figure(2);
clf;
voxels_all=(voxels+watervoxels)+(voxels>0).*1i+(watervoxels>0).*2i;
voxelSurf(voxels_all,true);

colormap([stonecolor;grasscolor;watercolor;trunkcolor;leavescolor]);
alphamap([1;0.5]);
axis([0 XX+1 0 YY+1 0 ZZ+1]);
light('Position',[128 0 256],'Style','local')


