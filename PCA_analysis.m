% prepare data
close all; clc;clear;

% fast = load("fast.mat");
% slow = load("slow.mat");
% fast = load("fast_temp.mat");
% slow = load("slow_temp.mat");

load("simulation_result_optimized");
%%
fast.acts_easy = new_result.acts{1};
fast.acts_ambi = new_result.acts{2};
fast.acts_misl = new_result.acts{3};
slow.acts_easy = new_result.acts{4};
slow.acts_ambi = new_result.acts{5};
slow.acts_misl = new_result.acts{6};

%% step 1/2 - align trials and split into left and right
cut_pre = 1500;
cut_post = 500;

model.num_neurons = 90;
model.angle_step = 360/model.num_neurons;
model.degrees = [-180:model.angle_step:176];
model.index_neuron1 = find(model.degrees==-100);
model.index_neuron2 = find(model.degrees== 20);
% exp.model = model;
% 
% % simulation related 
% exp.plot = 0;

% cut the time trace
fast_easy = preprocess(fast,"easy");
fast_ambi = preprocess(fast,"ambi");
fast_misl = preprocess(fast,"misl");
slow_easy = preprocess(slow,"easy");
slow_ambi = preprocess(slow,"ambi");
slow_misl = preprocess(slow,"misl");

% commit_fe = mean([fast_easy(:).commit_time]);
% commit_fa = mean([fast_ambi(:).commit_time]);
% commit_fm = mean([fast_misl(:).commit_time]);
%% plot the averaged result
[left,right] = avg_trials(fast_misl,model);

plot_acts(left.PMd_avg);
title('PMd');
plot_acts(left.M1_avg);
title('M1');

[left,right] = avg_trials(slow_misl,model);

plot_acts(left.PMd_avg);
title('PMd');
plot_acts(left.M1_avg);
title('M1');
%% step 3/4
% cell_mat = cell_matrix(fast_easy,fast_ambi,fast_misl,slow_easy,slow_ambi,slow_misl);
% cell_input = {fast_easy,fast_ambi,fast_misl};
cell_input = {fast_easy,fast_ambi,fast_misl,slow_easy,slow_ambi,slow_misl};

[cell_mat,region_record] = cell_matrix(cell_input,model);
[coeff,score,latent] = pca(cell_mat');
cal_varExp(latent);
%%
% plot across all neurons
close all;
mode = 'all';
dim_plot = 1;
[output_L,output_R] = PC_act(fast_easy,region_record,coeff,mode,model);
plot_pca(output_L,output_R,dim_plot);
[output_L,output_R] = PC_act(fast_ambi,region_record,coeff,mode,model);
plot_pca(output_L,output_R,dim_plot);
[output_L,output_R] = PC_act(fast_misl,region_record,coeff,mode,model);
plot_pca(output_L,output_R,dim_plot);
%%
close all;
% plot across each regions
mode = "PMd";
dim_plot = 1;
[output_L,output_R] = PC_act(fast_easy,region_record,coeff,mode,model);
plot_pca(output_L,output_R,dim_plot);
[output_L,output_R] = PC_act(fast_ambi,region_record,coeff,mode,model);
plot_pca(output_L,output_R,dim_plot);
[output_L,output_R] = PC_act(fast_misl,region_record,coeff,mode,model);
plot_pca(output_L,output_R,dim_plot);

%%
% plot across each regions
mode = "M1";
dim_plot = 3;
[output_L,output_R] = PC_act(fast_easy,region_record,coeff,mode,model);
plot_pca(output_L,output_R,dim_plot);
[output_L,output_R] = PC_act(fast_ambi,region_record,coeff,mode,model);
plot_pca(output_L,output_R,dim_plot);
[output_L,output_R] = PC_act(fast_misl,region_record,coeff,mode,model);
plot_pca(output_L,output_R,dim_plot);
%%
close all;
% plot across each regions
mode = "GPi";
dim_plot = 3;
[output_L,output_R] = PC_act(fast_easy,region_record,coeff,mode);
plot_pca(output_L,output_R,dim_plot);
[output_L,output_R] = PC_act(fast_ambi,region_record,coeff,mode);
plot_pca(output_L,output_R,dim_plot);
[output_L,output_R] = PC_act(fast_misl,region_record,coeff,mode);
plot_pca(output_L,output_R,dim_plot);
%%
close all;
figure;
imagesc(coeff(:, 1:10));  % Show first 10 principal components
colorbar;
xlabel('Principal Components');
ylabel('Time Steps (or Neurons)');
title('Loading Matrix Heatmap');
colormap(jet);  % Change color scheme

figure;
bar(coeff(:,1));  
xlabel('Neurons');
ylabel('Loading on PC1');
title('Neuron Contribution to PC1');
grid on;

figure;
scatter(coeff(:,1), coeff(:,2), 50, 'filled');
xlabel('PC1 Loadings');
ylabel('PC2 Loadings');
title('Neuron Loadings in PC1 vs PC2 Space');
grid on;

figure;
plot(cumsum(latent) / sum(latent) * 100, '-o');
xlabel('Number of Principal Components');
ylabel('Cumulative Variance Explained (%)');
title('Variance Explained by Principal Components');
grid on;

%% helper functions
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

% get the acts within the time window based on the commitment time
function new = preprocess(data,trial_type)
    if trial_type == "easy"
        var = data.acts_easy;
    elseif trial_type == "ambi"
        var = data.acts_ambi;
    elseif trial_type == "misl"
        var = data.acts_misl;
    end
    cut_pre = 1500;
    cut_post = 499;
    new = [];
    num_trial = size(var,2);
    for i = 1:num_trial
        temp = var(i);
        window = temp.commit_time-cut_pre:temp.commit_time+cut_post;
        % window = 1:3000; % change this line!!!
        temp.x_PFC = temp.x_PFC(:,window+1000);
        temp.x_PMd = temp.x_PMd(:,window+1000);
        temp.x_M1 = temp.x_M1(:,window+1000);
        temp.x_GPi = temp.x_GPi(:,window+1000);

        new = [new,temp];
    end
    
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



function [cell_mat,region_record] = cell_matrix(input,model)
    % % average over 250 trials, result in two groups: left right, 
    % % and three trials: easy ambi misl
    % for i = 1:length(input)
    %     temp = input{i};
    %     [temp_L{i},temp_R{i}] = avg_trials(temp);
    % end

    % play the trick, mirror the neurons
    for i = 1:length(input)
        temp = input{i};
        [temp_L{i},temp_R{i}] = avg_trials(temp,model);
        % [left,right] = avg_trials(temp,model);
        % % mirror the neuron
        % [left_updated, right_updated] = mirror_neurons(left, right,model);
        % temp_L{i} = left_updated;
        % temp_R{i} = right_updated;
    end

    % Change this after!!
    cell_mat = [];
    region_record = [];


    F_L = temp_L{1}.PFC_avg + temp_L{2}.PFC_avg + temp_L{3}.PFC_avg;
    F_R = temp_R{1}.PFC_avg + temp_R{2}.PFC_avg + temp_R{3}.PFC_avg;
    S_L = temp_L{4}.PFC_avg + temp_L{5}.PFC_avg + temp_L{6}.PFC_avg;
    S_R = temp_R{4}.PFC_avg + temp_R{5}.PFC_avg + temp_R{6}.PFC_avg;

    cell_mat = [cell_mat; F_L,F_R,S_L,S_R];
    % cell_mat = [cell_mat; F_L,F_R];
    num_neuron = size(F_R,1);
    region_record = [region_record;repmat("PFC",num_neuron,1)];

    F_L = temp_L{1}.PMd_avg + temp_L{2}.PMd_avg + temp_L{3}.PMd_avg;
    F_R = temp_R{1}.PMd_avg + temp_R{2}.PMd_avg + temp_R{3}.PMd_avg;
    S_L = temp_L{4}.PMd_avg + temp_L{5}.PMd_avg + temp_L{6}.PMd_avg;
    S_R = temp_R{4}.PMd_avg + temp_R{5}.PMd_avg + temp_R{6}.PMd_avg;
  

    cell_mat = [cell_mat;F_L,F_R,S_L,S_R];
    % cell_mat = [cell_mat; F_L,F_R];
    num_neuron = size(F_R,1);
    region_record = [region_record;repmat("PMd",num_neuron,1)];

    F_L = temp_L{1}.M1_avg + temp_L{2}.M1_avg + temp_L{3}.M1_avg;
    F_R = temp_R{1}.M1_avg + temp_R{2}.M1_avg + temp_R{3}.M1_avg;
    S_L = temp_L{4}.M1_avg + temp_L{5}.M1_avg + temp_L{6}.M1_avg;
    S_R = temp_R{4}.M1_avg + temp_R{5}.M1_avg + temp_R{6}.M1_avg;

    cell_mat = [cell_mat;F_L,F_R,S_L,S_R];
    % cell_mat = [cell_mat; F_L,F_R];
    num_neuron = size(F_R,1);
    region_record = [region_record;repmat("M1",num_neuron,1)];

    F_L = temp_L{1}.GPi_avg + temp_L{2}.GPi_avg + temp_L{3}.GPi_avg;
    F_R = temp_R{1}.GPi_avg + temp_R{2}.GPi_avg + temp_R{3}.GPi_avg;
    S_L = temp_L{4}.GPi_avg + temp_L{5}.GPi_avg + temp_L{6}.GPi_avg;
    S_R = temp_R{4}.GPi_avg + temp_R{5}.GPi_avg + temp_R{6}.GPi_avg;
    
    cell_mat = [cell_mat;F_L,F_R,S_L,S_R];
    % cell_mat = [cell_mat; F_L,F_R];
    num_neuron = size(F_R,1);
    region_record = [region_record;repmat("GPi",num_neuron,1)];

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
    for j = 1:num_trials
        temp = data(j);
        if temp.direction == 1
            result1.PFC_avg = result1.PFC_avg + temp.x_PFC;
            result1.PMd_avg = result1.PMd_avg + temp.x_PMd;
            result1.M1_avg = result1.M1_avg + temp.x_M1;
            result1.GPi_avg = result1.GPi_avg + temp.x_GPi;
        else
            result2.PFC_avg = result2.PFC_avg + temp.x_PFC;
            result2.PMd_avg = result2.PMd_avg + temp.x_PMd;
            result2.M1_avg = result2.M1_avg + temp.x_M1;
            result2.GPi_avg = result2.GPi_avg + temp.x_GPi;
        end
    end

    result1.PFC_avg = result1.PFC_avg/num_trials;
    result1.PMd_avg = result1.PMd_avg/num_trials;
    result1.M1_avg = result1.M1_avg/num_trials;
    result1.GPi_avg = result1.GPi_avg/num_trials;
    result2.PFC_avg = result2.PFC_avg/num_trials;
    result2.PMd_avg = result2.PMd_avg/num_trials;
    result2.M1_avg = result2.M1_avg/num_trials;
    result2.GPi_avg = result2.GPi_avg/num_trials;

    if result2.PFC_avg == 0
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
    elseif result1.PFC_avg == 0
        model = varargin{1};
        activity = result2.PFC_avg;
        flipped = flip_neurons_between(activity, model.index_neuron1, model.index_neuron2);
        result1.PFC_avg = flipped;

        activity = result2.PMd_avg;
        flipped = flip_neurons_between(activity, model.index_neuron1, model.index_neuron2);
        result1.PMd_avg = flipped;

        activity = result2.M1_avg;
        flipped = flip_neurons_between(activity, model.index_neuron1, model.index_neuron2);
        result1.M1_avg = flipped;
        
        activity = result2.GPi_avg;
        flipped = flip_neurons_between(activity, model.index_neuron1, model.index_neuron2);
        result1.GPi_avg = flipped;
    end
end

function plot_acts(neuron_activity)
figure;
[X, Y] = meshgrid(1:size(neuron_activity, 2), 1:size(neuron_activity, 1));
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


function plot_pca(output_L,output_R,dim,varargin)

if isempty(varargin)
    marked_time = 1500;
else
    marked_time = varargin{1};
end

if dim == 1
    % plot in 1D
    figure;
    subplot(2,4,1);
    plot(output_L(1,:));
    hold on;
    subplot(2,4,2);
    plot(output_L(2,:));
    hold on;
    subplot(2,4,3);
    plot(output_L(3,:));
    hold on;
    subplot(2,4,4);
    plot(output_L(4,:));

    subplot(2,4,5);
    plot(output_R(1,:));
    hold on;
    subplot(2,4,6);
    plot(output_R(2,:));
    hold on;
    subplot(2,4,7);
    plot(output_R(3,:));
     hold on;
    subplot(2,4,8);
    plot(output_R(4,:));
elseif dim == 2
    % plot in 2D: pc1 and pc2 dim
    figure;
    plot(output_L(1,:),output_L(2,:));
    hold on
    plot(output_R(1,:),output_R(2,:));
elseif dim == 3
    figure;
    plot3(output_L(1,:),output_L(2,:),output_L(3,:));
    hold on
    plot3(output_R(1,:),output_R(2,:),output_R(3,:));
    hold on
    % plot the commit point
    i = floor(marked_time);
    plot3(output_L(1,i), output_L(2,i),output_L(3,i), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
    plot3(output_R(1,i), output_R(2,i),output_R(3,i), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
end


end




% 
% function [left,right] = mirror_neurons(left, right, model)
%     % update right group
%     activity = left.PFC_avg;
%     flipped = flip_neurons_between(activity, model.index_neuron1, model.index_neuron2);
%     right.PFC_avg = (activity + right.PFC_avg)/2;
% 
%     activity = left.PMd_avg;
%     flipped = flip_neurons_between(activity, model.index_neuron1, model.index_neuron2);
%     right.PMd_avg = (activity + right.PMd_avg)/2;
% 
%     activity = left.M1_avg;
%     flipped = flip_neurons_between(activity, model.index_neuron1, model.index_neuron2);
%     right.M1_avg = (activity + right.M1_avg)/2;
% 
%     activity = left.GPi_avg;
%     flipped = flip_neurons_between(activity, model.index_neuron1, model.index_neuron2);
%     right.GPi_avg = (activity + right.GPi_avg)/2;
% 
%     % update left group
%     activity = right.PFC_avg;
%     flipped = flip_neurons_between(activity, model.index_neuron1, model.index_neuron2);
%     left.PFC_avg = (activity + left.PFC_avg)/2;
% 
%     activity = right.PMd_avg;
%     flipped = flip_neurons_between(activity, model.index_neuron1, model.index_neuron2);
%     left.PMd_avg = (activity + left.PMd_avg)/2;
% 
%     activity = right.M1_avg;
%     flipped = flip_neurons_between(activity, model.index_neuron1, model.index_neuron2);
%     left.M1_avg = (activity + left.M1_avg)/2;
% 
%     activity = right.GPi_avg;
%     flipped = flip_neurons_between(activity, model.index_neuron1, model.index_neuron2);
%     left.GPi_avg = (activity + left.GPi_avg)/2;
% 
%     % % visualize the result
%     % figure;
%     % subplot(1,2,1); imagesc(activity); title('Original Activity (Neuron x Time)');
%     % subplot(1,2,2); imagesc(flipped); title(sprintf('Flipped between neurons %d and %d', model.index_neuron1, model.index_neuron2));
% 
% end
