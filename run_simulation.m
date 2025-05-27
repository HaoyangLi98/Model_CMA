% output is used to evaluate, which should includes: decision time...
function output = run_simulation(x)
initial_pars;
opt_pars = parArr2struct(x,p);

% model
model.num_neurons = 90;
model.angle_step = 360/model.num_neurons;
model.degrees = [-180:model.angle_step:176];
model.index_neuron1 = find(model.degrees==-100);
model.index_neuron2 = find(model.degrees== 20);

Kd_pmd = kernel(opt_pars.Kd_pmd_1,opt_pars.Kd_pmd_2,opt_pars.Kd_pmd_3);
Kd_gpi = kernel(opt_pars.Kd_gpi_1,opt_pars.Kd_gpi_2,opt_pars.Kd_gpi_3);
model.weights_pmd = weights_matrix(model.num_neurons,Kd_pmd);
model.weights_gpi = weights_matrix(model.num_neurons,Kd_gpi);

model.timeSteps = [-1000:4000]'; % this tells how long to simulate, less than 0 means no evidence
model.opt_pars = opt_pars;
exp.model = model;

% simulation related 
exp.plot = 0;
exp.num_trials = 249;
%%
% load data
exp_input = load('./token_trials_modified.mat');

%%
% simulate three trial types
exp.urgency = 'fast';
% simulate begin
exp.trial_type = 'easy';
inp = input_conditioned(exp_input,exp);
acts = simulation_main(exp,inp);
output.acts{1} = acts;
output.exp{1} = exp;

exp.trial_type = 'ambi';
inp = input_conditioned(exp_input,exp);
acts = simulation_main(exp,inp);
output.acts{2} = acts;
output.exp{2} = exp;

exp.trial_type = 'misl';
inp = input_conditioned(exp_input,exp);
acts = simulation_main(exp,inp);
output.acts{3} = acts;
output.exp{3} = exp;

% simulate three trial types
exp.urgency = 'slow';
% simulate begin
exp.trial_type = 'easy';
inp = input_conditioned(exp_input,exp);
acts = simulation_main(exp,inp);
output.acts{4} = acts;
output.exp{4} = exp;

exp.trial_type = 'ambi';
inp = input_conditioned(exp_input,exp);
acts = simulation_main(exp,inp);
output.acts{5} = acts;
output.exp{5} = exp;

exp.trial_type = 'misl';
inp = input_conditioned(exp_input,exp);
acts = simulation_main(exp,inp);
output.acts{6} = acts;
output.exp{6} = exp;
end

%% other functions
function par = parArr2struct(x,p)
fields = fieldnames(p);
par = struct();
for i = 1:numel(fields)
    par.(fields{i}) = x(i);
end
end


function Kd = kernel(kappa,rho,width)


% kappa = 0.8;  % Scaling factor
% rho = 0.04;    % Offset parameter, this par is very sensitive
% width = 0.2;
% Define distance values for visualization
d = -4.5:0.1:4.6;  % Distance between neighboring cells

% Compute the Difference of Gaussians (DoG) kernel
gaussian1 = exp(-d.^2 / width) / sqrt(2 * pi);
Kd = kappa * gaussian1 - rho;

% Add 20% Gaussian noise (20% of the standard deviation of Kd)
noise = 0.1 * std(Kd) * randn(size(Kd));
%Kd = Kd + noise;

end


function weights = weights_matrix(num_neurons,kernel)
    % cal distance between neurons
    neuron_indices = (1:num_neurons)';
    
    distance_matrix = min(abs(neuron_indices - neuron_indices'), ...
                          num_neurons - abs(neuron_indices - neuron_indices'));
    % calculate weights
    weights = kernel((num_neurons/2+1) - distance_matrix); % par here should less than 0.1 to align with poune's range

end

function inp = input_conditioned(exp_input,exp)
inp = {};
if exp.trial_type == 'easy'
    inp{1} = exp_input.inp1_easy;
    inp{2} = exp_input.inp2_easy;
elseif exp.trial_type == 'ambi'
    inp{1} = exp_input.inp1_ambi;
    inp{2} = exp_input.inp2_ambi;
elseif exp.trial_type == 'misl'
    inp{1} = exp_input.inp1_misl;
    inp{2} = exp_input.inp2_misl;
else
    disp('tell which trial type');
    return
end
end