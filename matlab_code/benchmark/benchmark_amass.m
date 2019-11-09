clearvars; close all; clc; addpath(genpath(fileparts(mfilename('fullpath')))); 
%-------------------------------------------------------------------------%
%
%-------------------------------------------------------------------------%

[gt_set,res_set] = import_amass_sets('Experiment_1_Amass test set generalization'); 