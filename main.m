clc;clear;close all;

%% ini parallel
if isempty(gcp('nocreate'))
    parpool;
end
%% initial setting
% initial_pars;
% 
% x0 = pars2array(p);
load('optimized_result.mat');
x0 = bestever.x;
% sigma = [0.3*x0]'; % less value means more confidence about the initial setting
sigma = [
    0.002; 0.01; 0.02; 0.1;
    0.005; 0.02; 0.01;
    0.07; 0.05; 0.07; 0.03; 0.02;
    0.3; 0.02; 0.07;
    0.3; 0.02; 0.07;
    1.0; 0.2; 0.001; 0.5; 0.002;0.15
];

opts.EvalParallel = 'yes';
opts.LBounds = zeros(length(x0),1);
opts.PopSize = 30;
opts.MaxIter = 5;
opts.StopOnEqualFunctionValues = 3;
% opts.StopOnWarnings = 'no';

%%

[xmin, fmin, counteval, stopflag, out, bestever] = cmaes('score_function_parallel',x0,sigma,opts);

% [xmin, fmin, counteval, stopflag, out, bestever] = cmaes('score_function_parallel',x0,0.1,opts);

% close all;
plot_acts = 0;
plot_behaviour = 1;

% with new pars
new_result = run_simulation(bestever.x');

% with initial pars
old_result = run_simulation(x0);
% plot
plot_result(old_result,plot_acts,plot_behaviour);
old_score = cal_score(old_result);

plot_result(new_result,plot_acts,plot_behaviour);

new_score = cal_score(new_result);
%% save the optimized pars
save('optimized_result','bestever');
save('simulation_result_optimized','new_result');

save('x0','x0');
%% write the 