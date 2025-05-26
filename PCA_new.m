%% prepare data
close all; clc;clear;

data = load("simulation_result_optimized");


%% step 1 2 3
acts = data.new_result.acts;
exp = data.new_result.exp;

[cell_mat,region_info] = cell_matrix(acts,exp);

%% step 4
[coeff,score,latent] = pca(cell_mat');
cal_varExp(latent);
%% step 5 6
mode = 'all';

timeDomain_pca = timeDomain_acts(acts, exp, region_info, coeff, mode);
%%
plot_dim = 1;
plot_pca(timeDomain_pca,plot_dim);
%%
plot_dim = 3;
plot_pca(timeDomain_pca,plot_dim);
%%
function plot_pca(timeDomain_pca,plot_dim)
if plot_dim == 1
    plot_1d(timeDomain_pca);
else
    plot_3d(timeDomain_pca);
end
end



function plot_1d(timeDomain_pca)
figure;
[pc_counts,~] = size(timeDomain_pca.left{1,1});
% go through each pc
for i = 1:pc_counts
    subplot(1,pc_counts,i);
    % go through each trial
    for j = 1:3
        temp_L = timeDomain_pca.left{1,j};
        temp_L = temp_L(i,:);
        temp_R = timeDomain_pca.right{1,j};
        temp_R = temp_R(i,:);
        plot(temp_L,'--');
        plot(temp_R,'--');
        hold on
    end
    for j = 4:6
        temp_L = timeDomain_pca.left{1,j};
        temp_L = temp_L(i,:);
        temp_R = timeDomain_pca.right{1,j};
        temp_R = temp_R(i,:);
        plot(temp_L);
        plot(temp_R);
        hold on
    end
end
end



function plot_3d(timeDomain_pca)
output_L = timeDomain_pca.left;
output_R = timeDomain_pca.right;

figure;
for i = 1:3
    temp_l = output_L{1,i};
    plot3(temp_l(1,:),temp_l(2,:),temp_l(3,:),'--');
    hold on;

    temp_r = output_R{1,i};
    plot3(temp_r(1,:),temp_r(2,:),temp_r(3,:),'--');
    hold on;
end

for i = 4:6
    temp_l = output_L{1,i};
    plot3(temp_l(1,:),temp_l(2,:),temp_l(3,:));
    hold on;

    temp_r = output_R{1,i};
    plot3(temp_r(1,:),temp_r(2,:),temp_r(3,:));
    hold on;
end

% % plot the commit point
% i = floor(marked_time);
% plot3(output_L(1,i), output_L(2,i),output_L(3,i), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
% plot3(output_R(1,i), output_R(2,i),output_R(3,i), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
end


% main function for calculating the pca acts in time domain. extract the PC acts in time domain
% acts, exp:
% region_record: the regions each neuron from
% coeff: pca result
% mode: what region is based for calculation
% output: first 4 pc's acts for left and right group
function output = timeDomain_acts(acts, exp, region_info, coeff, mode)
n_conditions = length(acts);

for i = 1:n_conditions
    exp_temp = exp{1,i};
    act_temp = acts{1,i};
    urgency_type = exp_temp.urgency;
    trial_type = exp_temp.trial_type;
    [output.left{1,i},output.right{1,i}] = PC_act(act_temp,region_info,coeff,mode,exp_temp.model);
end
end



function [output_L,output_R] = PC_act(data,region_record,coeff,region,model)
    [temp_L,temp_R] = avg_trials(data,model);

    if region == "all"
        mat_L = [temp_L.PFC_avg;temp_L.PMd_avg;temp_L.M1_avg;temp_L.GPi_avg];
        mat_R = [temp_R.PFC_avg;temp_R.PMd_avg;temp_R.M1_avg;temp_R.GPi_avg];
        % mat_L = [temp_L.PFC_avg;temp_L.PMd_avg;temp_L.M1_avg];
        % mat_R = [temp_R.PFC_avg;temp_R.PMd_avg;temp_R.M1_avg];
        index = 1:length(region_record);
    elseif region == "PFC"
        mat_L = [temp_L.PFC_avg];
        mat_R = [temp_R.PFC_avg];
        index = find(region_record == "PFC");
    elseif region == "PMd"
        mat_L = [temp_L.PMd_avg];
        mat_R = [temp_R.PMd_avg];
        index = find(region_record == "PMd");
    elseif region == "M1"
        mat_L = [temp_L.M1_avg];
        mat_R = [temp_R.M1_avg];
        index = find(region_record == "M1");
    elseif region == "GPi"
        mat_L = [temp_L.GPi_avg];
        mat_R = [temp_R.GPi_avg];
        index = find(region_record == "GPi");
    end
   

    pc = [1,2,3,4];
    output_L = [];
    output_R = [];
    coeff = coeff(index,:);
    for i = 1:length(pc)
        output_L = [output_L; mean(mat_L.*coeff(:,pc(i)))];
        output_R = [output_R; mean(mat_R.*coeff(:,pc(i)))];
    end
end


% generate the cell matrix which is prepared for pca analysis later
% including trial_avg and mirror tricks
function [cell_mat,region_record] = cell_matrix(acts,exp)
    cut_post = 500;
    % for each condition, average over trials
    for i = 1:length(acts)
        model = exp{1,i}.model;
        temp_act = acts{i};
        temp_act = align_commitment(temp_act,cut_post);
        [temp_L{i},temp_R{i}] = avg_trials(temp_act,model);
    end

    cell_mat = [];
    region_record = [];

    % average over easy ambi misl
    F_L = average_across_conditions([temp_L{1}, temp_L{2}, temp_L{3}]);
    F_R = average_across_conditions([temp_R{1}, temp_R{2}, temp_R{3}]);
    S_L = average_across_conditions([temp_L{4}, temp_L{5}, temp_L{6}]);
    S_R = average_across_conditions([temp_R{4}, temp_R{5}, temp_R{6}]);

    % make the cell matrix
    F_L = [F_L.PFC_avg;F_L.PMd_avg;F_L.M1_avg;F_L.GPi_avg];
    F_R = [F_R.PFC_avg;F_R.PMd_avg;F_R.M1_avg;F_R.GPi_avg];
    S_L = [S_L.PFC_avg;S_L.PMd_avg;S_L.M1_avg;S_L.GPi_avg];
    S_R = [S_R.PFC_avg;S_R.PMd_avg;S_R.M1_avg;S_R.GPi_avg];

    cell_mat = [cell_mat; F_L,F_R,S_L,S_R];

    num_neuron = 90;
    region_record = [region_record;repmat("PFC",num_neuron,1)];
    region_record = [region_record;repmat("PMd",num_neuron,1)];
    region_record = [region_record;repmat("M1",num_neuron,1)];
    region_record = [region_record;repmat("GPi",num_neuron,1)];
    % 
    % F_L = temp_L{1}.PMd_avg + temp_L{2}.PMd_avg + temp_L{3}.PMd_avg;
    % F_R = temp_R{1}.PMd_avg + temp_R{2}.PMd_avg + temp_R{3}.PMd_avg;
    % S_L = temp_L{4}.PMd_avg + temp_L{5}.PMd_avg + temp_L{6}.PMd_avg;
    % S_R = temp_R{4}.PMd_avg + temp_R{5}.PMd_avg + temp_R{6}.PMd_avg;
    % 
    % 
    % cell_mat = [cell_mat;F_L,F_R,S_L,S_R];
    % % cell_mat = [cell_mat; F_L,F_R];
    % num_neuron = size(F_R,1);
    % region_record = [region_record;repmat("PMd",num_neuron,1)];
    % 
    % F_L = temp_L{1}.M1_avg + temp_L{2}.M1_avg + temp_L{3}.M1_avg;
    % F_R = temp_R{1}.M1_avg + temp_R{2}.M1_avg + temp_R{3}.M1_avg;
    % S_L = temp_L{4}.M1_avg + temp_L{5}.M1_avg + temp_L{6}.M1_avg;
    % S_R = temp_R{4}.M1_avg + temp_R{5}.M1_avg + temp_R{6}.M1_avg;
    % 
    % cell_mat = [cell_mat;F_L,F_R,S_L,S_R];
    % % cell_mat = [cell_mat; F_L,F_R];
    % num_neuron = size(F_R,1);
    % region_record = [region_record;repmat("M1",num_neuron,1)];
    % 
    % F_L = temp_L{1}.GPi_avg + temp_L{2}.GPi_avg + temp_L{3}.GPi_avg;
    % F_R = temp_R{1}.GPi_avg + temp_R{2}.GPi_avg + temp_R{3}.GPi_avg;
    % S_L = temp_L{4}.GPi_avg + temp_L{5}.GPi_avg + temp_L{6}.GPi_avg;
    % S_R = temp_R{4}.GPi_avg + temp_R{5}.GPi_avg + temp_R{6}.GPi_avg;
    % 
    % cell_mat = [cell_mat;F_L,F_R,S_L,S_R];
    % % cell_mat = [cell_mat; F_L,F_R];
    % num_neuron = size(F_R,1);
    % region_record = [region_record;repmat("GPi",num_neuron,1)];

end


function avg_all = average_across_conditions(aligned_structs)
% aligned_structs: 1x3 struct array (one per condition), each with .PFC_avg, .PMd_avg, ...
% region_fieldnames: e.g., {'PFC_avg', 'PMd_avg', 'M1_avg', 'GPi_avg'}

region_fieldnames = {'PFC_avg', 'PMd_avg', 'M1_avg', 'GPi_avg'};
n_cond = numel(aligned_structs);
n_regions = numel(region_fieldnames);
avg_all = struct();

for r = 1:n_regions
    field = region_fieldnames{r};

    % Get max time length across 3 conditions
    max_T = max(arrayfun(@(s) size(s.(field), 2), aligned_structs));
    n_neurons = size(aligned_structs(1).(field), 1);

    % Preallocate aligned array
    aligned = NaN(n_cond, n_neurons, max_T);

    for c = 1:n_cond
        X = aligned_structs(c).(field);  % [neurons × T]
        T = size(X, 2);
        aligned(c,:,end-T+1:end) = X;    % pad on the left
    end

    % Average across conditions
    avg_all.(field) = squeeze(nanmean(aligned, 1));  % [neurons × time]
end
end


% get the acts within the time window based on the commitment time
function new = align_commitment(data,cut_post)
  
    new = [];
    num_trial = size(data,2);
    for i = 1:num_trial
        temp = data(i);
        % window = temp.commit_time-cut_pre:temp.commit_time+cut_post;
        window = 1:temp.commit_time+cut_post;
        % window = 1:3000; % change this line!!!
        temp.x_PFC = temp.x_PFC(:,window+1000);
        temp.x_PMd = temp.x_PMd(:,window+1000);
        temp.x_M1 = temp.x_M1(:,window+1000);
        temp.x_GPi = temp.x_GPi(:,window+1000);

        new = [new,temp];
    end
    
end

% average over trials, if there is only one direction, do mirror
function [result1,result2] = avg_trials(data,varargin)
    result1.PFC_avg = 0;
    result1.PMd_avg = 0;
    result1.M1_avg = 0;
    result1.GPi_avg = 0;
    result2.PFC_avg = 0;
    result2.PMd_avg = 0;
    result2.M1_avg = 0;
    result2.GPi_avg = 0;

    num_trials = size(data,2);
    % for j = 1:num_trials
    %     temp = data(j);
    %     if temp.direction == 1
    %         result1.PFC_avg = average_by_commit(temp, 'x_PFC');
    %         result1.PMd_avg = result1.PMd_avg + temp.x_PMd;
    %         result1.M1_avg = result1.M1_avg + temp.x_M1;
    %         result1.GPi_avg = result1.GPi_avg + temp.x_GPi;
    %         % result1.PFC_avg = result1.PFC_avg + temp.x_PFC;
    %         % result1.PMd_avg = result1.PMd_avg + temp.x_PMd;
    %         % result1.M1_avg = result1.M1_avg + temp.x_M1;
    %         % result1.GPi_avg = result1.GPi_avg + temp.x_GPi;
    %     else
    %         result2.PFC_avg = result2.PFC_avg + temp.x_PFC;
    %         result2.PMd_avg = result2.PMd_avg + temp.x_PMd;
    %         result2.M1_avg = result2.M1_avg + temp.x_M1;
    %         result2.GPi_avg = result2.GPi_avg + temp.x_GPi;
    %     end
    % end
    
    % calculate the average
    result1.PFC_avg = average_by_commit(data, 'x_PFC');
    result1.PMd_avg = average_by_commit(data, 'x_PMd');
    result1.M1_avg = average_by_commit(data, 'x_M1');
    result1.GPi_avg = average_by_commit(data, 'x_GPi');

    % flip neuron - make the mirror
    model = varargin{1};
    activity = result1.PFC_avg;
    flipped = flip_neurons_between(activity, model.index_neuron1, model.index_neuron2);
    result2.PFC_avg = flipped;

    activity = result1.PMd_avg;
    flipped = flip_neurons_between(activity, model.index_neuron1, model.index_neuron2);
    result2.PMd_avg = flipped;

    activity = result1.M1_avg;
    flipped = flip_neurons_between(activity, model.index_neuron1, model.index_neuron2);
    result2.M1_avg = flipped;
    
    activity = result1.GPi_avg;
    flipped = flip_neurons_between(activity, model.index_neuron1, model.index_neuron2);
    result2.GPi_avg = flipped;

    % 
    % if result2.PFC_avg == 0
    %     model = varargin{1};
    %     activity = result1.PFC_avg;
    %     flipped = flip_neurons_between(activity, model.index_neuron1, model.index_neuron2);
    %     result2.PFC_avg = flipped;
    % 
    %     activity = result1.PMd_avg;
    %     flipped = flip_neurons_between(activity, model.index_neuron1, model.index_neuron2);
    %     result2.PMd_avg = flipped;
    % 
    %     activity = result1.M1_avg;
    %     flipped = flip_neurons_between(activity, model.index_neuron1, model.index_neuron2);
    %     result2.M1_avg = flipped;
    % 
    %     activity = result1.GPi_avg;
    %     flipped = flip_neurons_between(activity, model.index_neuron1, model.index_neuron2);
    %     result2.GPi_avg = flipped;
    % elseif result1.PFC_avg == 0
    %     model = varargin{1};
    %     activity = result2.PFC_avg;
    %     flipped = flip_neurons_between(activity, model.index_neuron1, model.index_neuron2);
    %     result1.PFC_avg = flipped;
    % 
    %     activity = result2.PMd_avg;
    %     flipped = flip_neurons_between(activity, model.index_neuron1, model.index_neuron2);
    %     result1.PMd_avg = flipped;
    % 
    %     activity = result2.M1_avg;
    %     flipped = flip_neurons_between(activity, model.index_neuron1, model.index_neuron2);
    %     result1.M1_avg = flipped;
    % 
    %     activity = result2.GPi_avg;
    %     flipped = flip_neurons_between(activity, model.index_neuron1, model.index_neuron2);
    %     result1.GPi_avg = flipped;
    % end
end


function avg_act = average_by_commit(data, region_field)
% data: struct array with fields .x_PFC (or others) and .commit_time
% region_field: name of the field to extract (e.g., 'x_PFC')

n_trials = numel(data);
n_neurons = size(data(1).(region_field), 1);
max_time = max(arrayfun(@(d) size(d.(region_field), 2), data));
% Preallocate [trials x neurons x time]
aligned = NaN(n_trials, n_neurons, max_time);

for i = 1:n_trials
    act = data(i).(region_field);           % [neurons x T_i]
    ct = size(act,2);               % where the end should align

    % Align: put act at the end, pad with NaN at the beginning
    aligned(i,:, (max_time-ct+1):max_time) = act;
end

% Average across trials (ignoring NaNs)
avg_act = squeeze(nanmean(aligned, 1));  % [neurons x time]
end

function flipped_activity = flip_neurons_between(activity, i, j)
    % activity: [N x T], each row = a neuron
    % i, j: indices of neurons between which the flip axis lies

    [N, T] = size(activity);

    % Compute center (can be fractional)
    flip_center = (i + j) / 2;

    % Original neuron indices
    idx = 1:N;

    % Compute offsets from the flip center
    offsets = idx - flip_center;

    % Mirror positions
    mirrored_pos = flip_center - offsets;

    % Round to nearest integer to find valid indices
    mirrored_idx = round(mirrored_pos);

    % Clamp to ensure indices stay within [1, N]
    mirrored_idx = max(min(mirrored_idx, N), 1);

    % Flip neuron order (rows), time stays the same (columns)
    flipped_activity = activity(mirrored_idx, :);
end


function cal_varExp(latent)
% Assume latent is a column vector of eigenvalues from PCA
% e.g., [coeff, score, latent] = pca(X);

% Calculate cumulative explained variance
explained = 100 * latent / sum(latent);
cumulativeExplained = cumsum(explained);

% Plot
figure;
plot(cumulativeExplained, '-ob', 'LineWidth', 2, ...
     'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'r', 'MarkerSize', 8);
xlabel('Components');
ylabel('Cumulative % variance explained');
title('Variance explained');
grid on;

% Optional: limit number of components shown
xlim([1 min(20, length(latent))]);
ylim([0 100]);
end