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
    full_model.VERT = full_model.VERT * 80;
    full_model.S_tri = calc_tri_areas(full_model);
    full_model.VERT = full_model.VERT - repmat(mean(full_model.VERT),full_model.n,1);
    
    
    
    
    for k=1:1 % four elevations
        
        null = full_model;
        
        if k>1
            null.VERT = null.VERT * rotate_x(50.0*(rand()-0.5));
        end
        
        % five random rotations
        n_cores = 4;
        angle_offset = rand() * 2*pi;
        angles = (1:n_cores) * 2*pi / n_cores + angle_offset;
        % angles = angles(randperm(length(angles)));
        % angles = angles(1:5)
        
        
        tic
        parfor ai=1:length(angles)
            
            fprintf('Shape %d/%d, view %d/%d...\n', i, length(shapes), (k-1)*5+ai, length(angles));
            
            a = angles(ai);

            S = null;
            %S.VERT = S.VERT * [1 0 0 ; 0 0 1 ; 0 1 0];
%             coef = 20 / range(S.VERT(:,1));
            coef = 0.15*sqrt(sum(S.S_tri)) / max(range(S.VERT));
            S.VERT = coef.*S.VERT;
            
            %S.VERT = S.VERT * [cos(a) -sin(a) 0 ; sin(a) cos(a) 0 ; 0 0 1];
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
            parsave(out_fname, M.gt);
            
            fig = figure('visible', 'off');
            subplot(121), imagesc(depth), axis equal, colormap(gray), colorbar, title(sprintf('angle %.4f',a)), axis image
            subplot(122), plot_mesh(M), axis off; view([90 -90]); shading faceted
            fig_fname = sprintf('%s/%s.range%d.png', out_dir, nm, (k-1)*5+ai);
            saveas(fig, fig_fname);
            autocrop_image(fig_fname, true);
            
        end % next rotation
        toc
    end % next elevation
end % next shape
