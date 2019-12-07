function [moved] = compute_icp(moving,fixed,extrapolate)
fixed = pointCloud(fixed); moving = pointCloud(moving);
tform = pcregistericp(moving,fixed,'Extrapolate',extrapolate);
moved = pctransform(moving,tform); moved = moved.Location;
end
