clc;clear;close all;


%% ini parallel
if isempty(gcp('nocreate'))
    parpool;
end
%% initial setting
initial_pars;

x0 = pars2array(p);
% load('optimized_result.mat');
% x0 = bestever.x;
% sigma = [0.3*x0]'; % less value means more confidence about the initial setting

% load('x0.mat');
sigma = [
    0.002; 0.01; 0.02; 0.1;
    0.005; 0.02; 0.01;
    0.07; 0.05; 0.07; 0.03; 0.02;
    0.3; 0.02; 0.07;
    0.3; 0.02; 0.07;
    1.0; 0.2; 0.001; 0.5; 0.002
];

opts.EvalParallel = 'yes';
opts.LBounds = zeros(length(x0),1);
% opts.LBounds(end,1) = 0.2; % set the lbound for noise

opts.PopSize = 20;
opts.MaxIter = 100;
% opts.StopOnEqualFunctionValues = 1;
opts.StopFitness  = -2200;
% opts.StopOnWarnings = 'no';

%%

[xmin, fmin, counteval, stopflag, out, bestever] = cmaes('score_function_parallel',x0,sigma,opts);

% [xmin, fmin, counteval, stopflag, out, bestever] = cmaes('score_function_parallel',x0,0.1,opts);
%%
close all;
plot_acts = 1;
plot_behaviour = 1;

% with initial pars
old_result = run_simulation(x0);
% plot
plot_result(old_result,plot_acts,plot_behaviour);
old_score = cal_score(old_result);

% with new pars
new_result = run_simulation(bestever.x');


plot_result(new_result,plot_acts,plot_behaviour);

new_score = cal_score(new_result);
%% save the optimized pars
save('best_pars','xmin', 'fmin', 'counteval', 'stopflag', 'out', 'bestever');
save('simulation_result_optimized','new_result','-v7.3');

save('x0','x0');
%% write the 
close all
load("outcmaesxrecentbest.dat");
num_pars = length(x0);

idx = 63;

x0_mid = outcmaesxrecentbest(idx,end-num_pars+1:end);
result_mid = run_simulation(x0_mid);


plot_result(result_mid,plot_acts,plot_behaviour);
score_mid = cal_score(result_mid);

%%
test_data = load('optimized_result.mat');
x0_temp = test_data.bestever.x;
result_temp = run_simulation(x0_temp);

plot_result(result_temp,1,1);
score_temp = cal_score(result_temp);

