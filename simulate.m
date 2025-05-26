% this function simulate the system for defined steps
% This version is designed for distributed network based on previous model
% loop through each time step

function System = simulate(timeSteps, inp, exp)

if isfield(exp,'plot_acts_single') == 0
    exp.plot_acts_single = 1;
end


% model parameters
model = exp.model;
par = model.opt_pars;
B_PFC = 10;
B_PMd = 10;
B_M1 = 10;
B_GPi = 20;
L_PFC = 0.01;
L_PMd = 0.01;
L_M1 = 0.01;
L_GPi = 0;

% initialize neural acts
steps = length(timeSteps);
num_neurons = model.num_neurons; % num of neurons for each layer
if isfield(exp,'ini')
    ini = exp.ini;
else
    ini.PFC = zeros(num_neurons, 1);
    ini.PMd = zeros(num_neurons, 1);
    ini.M1 = zeros(num_neurons, 1);
    ini.GPi = zeros(num_neurons, 1);
end

x_PFC = [ini.PFC zeros(num_neurons,steps)];
x_PMd = [ini.PMd zeros(num_neurons,steps)];
x_M1 = [ini.M1 zeros(num_neurons,steps)];
x_GPi = [ini.GPi zeros(num_neurons,steps)];

% detect commitment
commit_time = -1;
max_neuron = -1;
for i = 1:steps

    % handle the case: simulation time > evidence
    if i > size(inp,2)
        input = inp(:,end);
    else
        input = inp(:,i);
    end
    
    % urgency grows when token jumps (after experiment start)
    if timeSteps(i,1) < 0
        u = 0.3*exp.phi;
    else
        u = U(timeSteps(i,1),exp.A,exp.u0_base,exp.phi);
    end

    % update for single timestep
    model.currentStep = i;

    % PFc layer
    e_pfc = E_PFC(input,par);
    x_PFC(:,i+1) = dxdt(L_PFC, B_PFC, e_pfc, 0, x_PFC(:,i)) + x_PFC(:,i);

    % PMd layer
    e_pmd = E_PMd(x_PMd(:,i), x_PFC(:,i), x_M1(:,i), model.weights_pmd, par);
    i_pmd = I_PMd(x_PMd(:,i), x_GPi(:,i), model.weights_pmd, par);
    x_PMd(:,i+1) = dxdt(L_PMd, B_PMd, e_pmd, i_pmd, x_PMd(:,i)) + x_PMd(:,i);

    % M1 layer
    e_m1 = E_M1(x_M1(:,i), x_PMd(:,i), par);
    i_m1 = I_M1(x_M1(:,i), par);
    x_M1(:,i+1) = dxdt(L_M1, B_M1, e_m1, i_m1, x_M1(:,i)) + x_M1(:,i);

    % GPi layer
    [commit,index,e_gpi] = E_GPi(x_GPi(:,i), x_PMd(:,i), model.weights_gpi, par);
    i_gpi = I_GPi(x_GPi(:,i), x_PMd(:,i), u, model.weights_gpi, par);
    x_GPi(:,i+1) = dxdt(L_GPi, B_GPi, e_gpi, i_gpi, x_GPi(:,i)) + x_GPi(:,i);

    % Upper and lower bound
    x_PFC(:,i+1) = max(0,min(B_PFC,x_PFC(:,i+1)));
    x_PMd(:,i+1) = max(0,min(B_PMd,x_PMd(:,i+1)));
    x_M1(:,i+1) = max(0,min(B_M1,x_M1(:,i+1)));
    x_GPi(:,i+1) = max(0,min(B_GPi,x_GPi(:,i+1)));

    % when commitment
    if commit == 1 && commit_time == -1
        commit_time = timeSteps(i);
        max_neuron = index;
    end
end

% store activities during time
System = struct('x_PFC',x_PFC,'x_PMd',x_PMd,'x_M1',x_M1,'x_GPi',x_GPi,'inp',inp,'commit_time',commit_time,'max_neuron',max_neuron);


if exp.plot_acts_single == 1
    plot_3D(System);
end
    
end


%% helper functions
function output = U(t,A,u0_base,phi) 
output = A * t + u0_base*phi; % need to add inter-trial noise later
end

function act=dxdt(L, B, E_i, I_i, x_i) 
act = -L * x_i + (B - x_i) .* E_i - x_i .* I_i;
end

function act = E_PFC(diff,par) 
act = par.E_PFC_w * max((1 + diff),0) + generateNoise();
end

function act = E_PMd(x_PMd, x_PFC, x_M1, weights, par)
    % exc: only positive weights are used 
    weights = max(0, weights); 
    lateral_term = sum(weights .* sigmoid(x_PMd,0.5,7,0), 1)'; 
    % lateral_term = 0.01*sigmoid(x_PMd);
    % final output: 1. from PFC 2. from itself and lateral
    % act = 0.02 * x_PFC + lateral_term; % this create a positive loop
    act = par.E_PMd_w1 * x_PFC + par.E_PMd_w2*sigmoid(x_M1,1,7,0) + lateral_term + generateNoise(); % this create a positive loop
end


function act = I_PMd(x_PMd, x_GPi, weights, par)

    % cal the lateral term
    weights = max(0, -weights); % rectified
    lateral_term = sum(weights .* sigmoid(x_PMd,0.5,7,0), 1)'; 
    % lateral_term = 0.03*sigmoid(x_PMd);
    % final output: 1. from itself and lateral
    % act = lateral_term;
    act = lateral_term + par.I_PMd_w*sigmoid(x_GPi,1,5,0.01);
    % act = lateral_term;
end


function act = E_M1(x_M1, x_PMd, par)
    % exc: only positive weights are used 
    % weights = max(0, weights); 
    %lateral_term = sum(weights .* sigmoid(x_M1,1,7,0), 1)'; 
    lateral_term = par.E_M1_w1*sigmoid(x_M1,1,7,0);
    % final output: 1. from PFC 2. from itself and lateral
    act = par.E_M1_w2 * sigmoid(x_PMd,0.5,7,0) + lateral_term + generateNoise(); % this create a positive loop
end


function act = I_M1(x_M1,par)

    % cal the lateral term
    % weights = max(0, -weights); % rectified
    % lateral_term = sum(weights .* sigmoid(x_M1,1,7,0), 1)'; 
    lateral_term = par.I_M1_w*sigmoid(x_M1,1,7,0);
    % final output: 1. from itself and lateral
    act = lateral_term;
end

function [commit,max_neuron,act] = E_GPi(x_GPi, x_PMd, weights, par)
    threshold = par.gpi_threshold;
    weights = max(0, weights); 
    lateral_term = sum(weights .* sigmoid(x_GPi,1,5,0.01), 1)'; 
    diff = max(x_PMd-x_PMd')';
    act = par.E_GPi_w1*sigmoid(diff,2,threshold,0.2) + par.E_GPi_w2*lateral_term + generateNoise();

    if max(diff)>threshold
        [~,max_neuron] = max(diff);
        commit = 1;
    else
        commit = 0;
        max_neuron = -1;
    end
end

function act = I_GPi(x_GPi, x_PMd, u, weights, par)
    threshold = par.gpi_threshold;
    weights = max(0, -weights); % rectified
    lateral_term = sum(weights .* sigmoid(x_GPi,1,5,0.01), 1)'; 
    diff = max(x_PMd'-x_PMd)';
    % act =  lateral_term + 0.05 * u;
    act = par.I_GPi_w1*sigmoid(diff,2,threshold,0.2) + par.I_GPi_w2*lateral_term + par.I_GPi_w3 * u;
end

function output = sigmoid(x,a,b,c)
% three pars a b c: a slope b central c y-offset. add c will remove the
% inh
output = 1 ./ (1 + exp(-a * (x - b))) + c;
end

function output = generateNoise()
% output = 0; % !!! change this after
output = (0.02 * (1 + 2 * randn()))^2;
end


function plot_3D(System)
figure;

% cutoff = 300;
% neuron_activity = System.x_GPi(:,cutoff:end);
% [X, Y] = meshgrid(1:size(neuron_activity, 2), 1:size(neuron_activity, 1));
% surf(X, Y, neuron_activity, 'EdgeColor', 'none'); % Smooth heatmap
% colormap(jet); % Use 'hot', 'parula', 'jet', etc.
% colorbar; % Show color scale
% title('3D Heat Map');
% xlabel('X-axis');
% ylabel('Y-axis');
% zlabel('Heat Intensity');
% %caxis([0,5]);
% view(3); % Set 3D perspective


neuron_activity = System.x_PMd;
[X, Y] = meshgrid(1:size(neuron_activity, 2), 1:size(neuron_activity, 1));
subplot(1,3,1)
surf(X, Y, neuron_activity, 'EdgeColor', 'none'); % Smooth heatmap
colormap(jet); % Use 'hot', 'parula', 'jet', etc.
colorbar; % Show color scale
title('3D Heat Map');
xlabel('X-axis');
ylabel('Y-axis');
zlabel('Heat Intensity');
%caxis([0,5]);
view(3); % Set 3D perspective

neuron_activity = System.x_M1;
[X, Y] = meshgrid(1:size(neuron_activity, 2), 1:size(neuron_activity, 1));
subplot(1,3,2)
surf(X, Y, neuron_activity, 'EdgeColor', 'none'); % Smooth heatmap
colormap(jet); % Use 'hot', 'parula', 'jet', etc.
colorbar; % Show color scale
title('3D Heat Map');
xlabel('X-axis');
ylabel('Y-axis');
zlabel('Heat Intensity');
%caxis([0,5]);
view(3); % Set 3D perspective

neuron_activity = System.x_GPi;
[X, Y] = meshgrid(1:size(neuron_activity, 2), 1:size(neuron_activity, 1));
subplot(1,3,3)
surf(X, Y, neuron_activity, 'EdgeColor', 'none'); % Smooth heatmap
colormap(jet); % Use 'hot', 'parula', 'jet', etc.
colorbar; % Show color scale
title('3D Heat Map');
xlabel('X-axis');
ylabel('Y-axis');
zlabel('Heat Intensity');
%caxis([0,5]);
view(3); % Set 3D perspective

end