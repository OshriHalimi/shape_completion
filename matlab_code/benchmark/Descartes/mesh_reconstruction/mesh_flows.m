function [ M,hist ] = mesh_flows( M,meth,opt,oplt_opt)

% Handle Defaults
o.lap_type = 'uniform';
o.lambda = 0.2;
o.mu = 0.15; % For Taubin flow
o.n_iter = 10;
o.to_show = 1;
o.record = 0;
o.follow = 1; 
o.replot = 0; 
o.pause = 0.03;
o.disp_func = @visualize;
o.face_color = 'banana'; 
o.frame_rate = 10;
o.keep_hist = 0;
o = mergestruct(o,opt,1);

hist = [];
% Initialization
if o.to_show
    oplt.new_fig = 0; 
    if o.replot 
        oplt.do_clf = 1; 
    else
        oplt.do_clf = 2;
    end
    oplt.clr_bar = 0; oplt.S.EdgeColor = [0,0,0];
    oplt.limits = 0; 
    C = uniclr(o.face_color,M.Nf);
    if exist('oplt_opt','var')
        oplt = mergestruct(oplt,oplt_opt,2); % Add in override
    end
    if o.record
        video_init(usprintf('%s %s on %s weights',M.name,meth,o.lap_type),o.frame_rate);
    end
end

update_func = str2func(meth);
if o.keep_hist
    hist = [hist , M ];
end
% TODO: Some of the computation is redundant - fix it
for i = 1:o.n_iter
    M = update_func(M,o);
    if o.keep_hist
        hist = [hist , M ];
    end
    if o.to_show
            
            if ~o.replot && exist('p','var') % For first iteration 
                pause(o.pause); 
                p.Vertices = M.v; 
                title(uplw(sprintf('%s %s on %s weights :: I-%d',M.name,meth,o.lap_type,i)));
                drawnow;
            else
                M.plt.title = (uplw(sprintf('%s %s on %s weights :: I-%d',M.name,meth,o.lap_type,i)));
                p = o.disp_func(M,C,oplt); 
                if o.follow
                    ylim manual; xlim manual; zlim manual;
                end
%                 add_xyz_axis(); grid on; 
            end
        if o.record
            video_append();
        end
    end
end
if o.to_show && o.record
    video_close();
end

end
%-------------------------------------------------------------------------%
%                           Flows Implementation
%-------------------------------------------------------------------------%
function [M] = spectra_truncation(M,~)
persistent V i v; % TODO - Get rid of the persistent
if isempty(V)
    opt.area_type = 'none';
    opt.lap_type = 'uniform';
    opt.skip_first_x = 0;
    i=3;
    [V,~] = mesh_laplacian_eigs(M,M.Nv,opt);
    v = M.v;
    %     D = sparse(diag(D));
end
V2 = V(:,1:i); i = i+1; 
M.v = V2*V2.'*v; % TODO - Add stop condition 
end
function [M] = mean_curvature(M,o)
if ~isfield(M.plt,'L')
    L = mesh_laplacian(M,o.lap_type); 
    % Diagonal must be negative - all off diagonal - positive
    if L(1,1) > 0
        L = -L; 
    end
    M.plt.L = L; % TODO - Change this to 'c'
else
    L = M.plt.L; 
end
% https://www.alecjacobson.com/weblog/?tag=mean-curvature-flow
A = massmatrix(M,'barycentric');
M.v = (A-o.lambda*L)\(A*M.v);
% Normalization - see post
M.v = M.v/sqrt(sum(doublearea(M.v,M.f))*0.5); % Transform to Unit Area
M.v = M.v - mesh_centroid(M); % Cancel translations 
end
function [M] = skeleton(M,o)
L = mesh_laplacian(M,o.lap_type); %Do we need to compute this again?
M.v = M.v - o.lambda *L*M.v; % TODO - Resolve odd discrepency with uniform mode
end
function [M] = taubin(M,o)
L = mesh_laplacian(M,o.lap_type);
M.v = M.v - o.lambda * normr(L*M.v);
L = mesh_laplacian(M,o.lap_type);
M.v = M.v + o.mu * normr(L*M.v);
end
function [M] = constant(M,o)
M = M.add_vertex_normals();
M.v = M.v - o.lambda * M.vn;
% [~,H] = mesh_curvature(M,'meyer');
% M.v = M.v + o.lambda * sign(H).* normr(L*M.v);
end
