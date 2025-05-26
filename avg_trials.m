
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
