%%
clc
close all
clear all

addpath('./tools/')

shapes_dir = './data/';
out_dir = './generated/';

shapes = dir(sprintf('%s/*.off', shapes_dir));
for i=1:length(shapes)
    
    [~,nm,~] = fileparts(shapes(i).name);
    
    full_model = load_off(sprintf('%s/%s', shapes_dir, shapes(i).name));
    full_model.S_tri = calc_tri_areas(full_model);
    full_model.VERT = full_model.VERT - repmat(mean(full_model.VERT),full_model.n,1);
    
    for k=1:4 % four elevations
        
        null = full_model;
        
        if k>1
            null.VERT = null.VERT * rotate_x(50.0*(rand()-0.5));
        end
        
        % five random rotations
        angles = linspace(0,2*pi,13);
        angles = angles(1:end-1);
        angles = angles(randperm(length(angles)));
        angles = angles(1:5);
        
        for ai=1:length(angles)
            
            fprintf('Shape %d/%d, view %d/%d...\n', i, length(shapes), (k-1)*5+ai, 4*length(angles));
            
            a = angles(ai);

            S = null;
            S.VERT = S.VERT * [1 0 0 ; 0 0 -1 ; 0 1 0];
%             coef = 20 / range(S.VERT(:,1));
            coef = 0.3*sqrt(sum(S.S_tri)) / max(range(S.VERT));
            S.VERT = coef.*S.VERT;
            
            S.VERT = S.VERT * [cos(a) 0 -sin(a) ; 0 1 0 ; sin(a) 0 cos(a)];
            
            [M, depth, matches] = create_rangemap(S, 95, 95, 3+randi(30));
            M.gt = matches;
            
            M.VERT = M.VERT - repmat(mean(M.VERT),M.n,1);
            M.VERT = M.VERT ./ coef;
            M = cleanup(M);
            
            if M.m<100
                continue;
            end
            
            out_fname = sprintf('%s/%s.range%d.mat', out_dir, nm, (k-1)*5+ai);
            save(out_fname, 'M', 'depth');
            
            fig = figure;
            subplot(121), imagesc(depth), axis equal, colormap(gray), colorbar, title(sprintf('angle %.4f',a)), axis image
            subplot(122), plot_mesh(M), axis off; view([90 -90]); shading faceted
            fig_fname = sprintf('%s/%s.range%d.png', out_dir, nm, (k-1)*5+ai);
            saveas(fig, fig_fname);
            autocrop_image(fig_fname, true);
            
        end % next rotation
    end % next elevation
end % next shape
