function [W,A] = cotlp_matrix(filename)
%
% Compute the Laplace-Beltrami matrix from mesh by cot scheme
%
% INPUTS
%  filename:  off file of triangle mesh.
%
% OUTPUTS
%  W: weight matrix 
%  A: area weight per vertex, the Laplace-Beltrami matrix = diag(1./ A) * W

echo off all
if nargin < 1
    error('Too few input arguments');	 
end
addpath(up_script_dir(0,'/cotlp'));
[II,JJ,SS,A] = cotlpmatrix(filename);
W=sparse(II, JJ, SS);
A = sparse(1:length(A), 1:length(A), A);


