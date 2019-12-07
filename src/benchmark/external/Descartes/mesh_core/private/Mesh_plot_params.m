function [M] = Mesh_plot_params(M)

% Figure Propertiesz
M.plt.new_fig = 1;
M.plt.do_clf = 0; 
M.plt.limits = 1; 

M.plt.clr_bar = 0;
M.plt.clr_map.name = 'super_jet'; % 'viridis' 'magma' 'plasma' 'redblue' 'inferno'
M.plt.clr_map.scale = -1; % Compute default scales
M.plt.clr_map.trunc = 'none'; % none sym outlier
M.plt.clr_map.invert = 0; 

% Edge plot 
M.plt.S.EdgeColor = [0,0,0]; 
M.plt.S.EdgeAlpha = 1; 
M.plt.S.LineWidth = 1; 
M.plt.S.LineStyle = 'none'; 

% Default Title 
if strcmp(M.name,'unknown')
    M.plt.title = ''; 
else
    M.plt.title = M.name; 
end

% Light & Angle 
M.plt.light = 1; 

% cleaned_name = regexprep(M.name ,'-?\d+$',''); 
cleaned_name = regexprep(M.name ,'-\d+.*|-[A-Za-z]{1,2}\d+.*|\d+',''); 
switch lower(cleaned_name)
    case {'unit-disk','rand-unit-disk','armadillo'}
        M.plt.disp_ang = [-180,-90];
    case {'dragon','smiley face','michael1'}
        M.plt.disp_ang = [0,90];
    case {'unit-sphere','spherocylinder'}
        M.plt.disp_ang = [0,0];
    case {'beetle'}
        M.plt.disp_ang = [0,180]; 
    case {'seashell'}
        M.plt.disp_ang = [-150,10]; 
    case {'bunny'}
        M.plt.disp_ang = [-6,28]; 
    case {'skeleton hand1'}
        M.plt.disp_ang = [90,-90]; 
    case {'horse'}
        M.plt.disp_ang = [-48,37];
    otherwise
        M.plt.disp_ang = [90,-90];
end
M.plt.post_disp_ang = []; % To adjust a different light angle
% Face normal Plot
[~,R] = M.box_bounds(); 
M.plt.normal_scaling = R/75;
M.plt.normal_clr =[];


end

