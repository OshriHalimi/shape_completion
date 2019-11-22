function [W,A,h] = symmshlp_matrix(filename, opt)
%
% Compute the symmetric Laplace-Beltrami matrix from mesh
%
% INPUTS
%  filename:  off file of triangle mesh.
%  opt.htype: the way to compute the parameter h. h = hs * neighborhoodsize
%             if htype = 'ddr' (data driven); h = hs if hytpe = 'psp' (pre-specify)
%             Default : 'ddr'
%  opt.hs:    the scaling factor that scales the neighborhood size to the
%             parameter h	where h^2 = 4t.
%             Default: 2, must > 0
%  opt.rho:   The cut-off for Gaussion function evaluation. 
%             Default: 3, must > 0
%  opt.dtype: the way to compute the distance 
%             dtype = 'euclidean' or 'geodesic';
%             Default : 'euclidean'

%
% OUTPUTS
%  W: symmetric weight matrix 
%  A: area weight per vertex, the Laplace matrix = diag(1./ A) * W 
%	h: Gaussian width: h^2 = 4t 


if nargin < 1
    error('Too few input arguments');	 
elseif nargin < 2
	opt.hs = 2;
	opt.rho = 3;
	opt.htype = 'ddr';
	opt.dtype = 'euclidean';
end
opt=parse_opt(opt);

if opt.hs <= 0 || opt.rho <= 0
	error('Invalid values in opt');
end

addpath(up_script_dir(0,'/cotlp'));
[II,JJ,SS,A,h] = symmshlpmatrix(filename, opt);
W=sparse(II, JJ, SS);
A = sparse(1:length(A), 1:length(A), A);

% Parsing Option.
function o = parse_opt(o)
if ~isfield(o,'hs')
    o.hs = 2; 
end
if ~isfield(o,'rho')
    o.rho = 3; 
end
if ~isfield(o,'htype')
    o.htype = 'ddr'; 
end
if ~isfield(o,'dtype')
    o.dtype = 'euclidean'; 
end

