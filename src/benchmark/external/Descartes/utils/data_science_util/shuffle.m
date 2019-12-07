 function v=shuffle(v) %TODO: Make this function more general. 
 if size(v,2)==1 || size(v,1)==1
     v=v(randperm(length(v)));
 else
     v = v(randperm(size(v,1)),:); % Permute rows
 end

