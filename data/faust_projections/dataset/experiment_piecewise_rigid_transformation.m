load('tr_reg_006.mat')
scatter3(full_shape(:,1),full_shape(:,2),full_shape(:,3),'r','filled'); axis equal
hold
ind_hands = full_shape(:,3)>0.3 & full_shape(:,1)>0;
hand = full_shape(ind_hands,:);
scatter3(hand(:,1),hand(:,2),hand(:,3),'g','filled'); axis equal
rot_axis = [0.17, 0.3,0.3];
scatter3(rot_axis(1),rot_axis(2),rot_axis(3),500,'b','filled'); axis equal
hand_relative = hand - rot_axis;
rot_direction = [1,0,0];
rotm = axang2rotm([rot_direction,-pi/4]);
hand_relative = (rotm*hand_relative')';
hand = hand_relative + rot_axis;
scatter3(hand(:,1),hand(:,2),hand(:,3),'k','filled'); axis equal
full_shape(ind_hands,:) = hand;
figure
scatter3(full_shape(:,1),full_shape(:,2),full_shape(:,3),'b','filled'); axis equal
