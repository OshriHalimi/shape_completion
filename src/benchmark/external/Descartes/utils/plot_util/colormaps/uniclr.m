function [clrmap] = uniclr(clr,N,clr2,ids)
if ~exist('N','var'); N = 1; end
if isnumeric(clr)
    clrmap = repmat(clr,N,1); 
    return;
end
clr = lower(clr);

switch clr
    case 'tot_rand'
        clrmap = rand(N,3);
    case 'tot_rand_purple'
        clrmap = hsv2rgb([randi([200,250],N,1) randi([200,255],N,2)]./255); 
    case 'tot_rand_blue'
        clrmap = hsv2rgb([randi([120,180],N,1) randi([200,255],N,2)]./255); 
    case 'tot_rand_green'
        clrmap = hsv2rgb([randi([60,120],N,1) randi([200,255],N,2)]./255); 
    case 'tot_rand_red'
        clrmap = hsv2rgb([randi([0,20],N,1) randi([200,255],N,2)]./255); 
    otherwise
        clrval = pick_clrval(clr);
        clrmap = repmat(clrval,N,1);
        if exist('clr2','var') && exist('ids','var') && ~isempty(ids)
            clr2 = lower(clr2);
            clrval2 = pick_clrval(clr2);
            clrmap2 = repmat(clrval2,numel(ids),1);
            clrmap(ids,:) = clrmap2;
        end
end
end

function [V] = pick_clrval(C)
switch C
    case {'o','orange'}
        V = [253,106,2]./255; 
    case {'b','blue'}
        V = [0,0,1];
    case {'k','black'}
        V = [0,0,0];
    case {'r','red'}
        V = [1,0,0];
    case {'t','turquoise'}
        V = [64,224,208]./255;
    case {'w','white'}
        V = [1,1,1];
    case {'c', 'cyan'}
        V = [0,1,1];
    case {'m','magenta'}
        V = [1,0,1];
    case {'g','green'}
        V = [0,1,0];
    case {'banana'}
        V = [254,240,152]./255; 
    case {'y','yellow'}
        V = [1,1,0];
    case {'pink'}
        V = [254,127,156]./255;
    case {'rand'}
        V = rand(1,3);
    case {'p','purple'}
        V = [135,31,120]./255; 
    case {'grey'}
        V = [202,204,206]./255;
    case {'gold'}
         V = [255 228  58]/255;
    case {'nblue'}
        V = [0.2 0.3 0.8]; 
    case {'teal'}
        V = [144 216 196]/255;
    case {'silver'}
        V = [192 192 192]/255; 
    case {'dark_pink'}
        V = [255 0 102]/255; 
    case {'beige'}
        V = [254,251,234]/255;
    case {'light blue'}
        V = [179,207,221]/255;
    otherwise
        error('Unimplemented color');
end
end