% two critera: for each condition, dt within the interval -100 (-600 in total)
% for each trial, max neuron within the range -300/number of trials 
function score = cal_score(data)
% score = randn*10;
score = 0;
num_conditions = 6;
num_trials = data.exp{1,1}.num_trials;
credit = 300/num_trials;
% go through 6 conditions
for i = 1:num_conditions
    max_neuron = [data.acts{1,i}.max_neuron];
    commit_time = [data.acts{1,i}.commit_time];
    urgency_type = data.exp{1,i}.urgency;
    trial_type = data.exp{1,i}.trial_type;
    if urgency_type == "fast" && trial_type == "easy"
        target_time = 600;
    elseif urgency_type == "fast" && trial_type == "ambi"
        target_time = 1000;
    elseif urgency_type == "fast" && trial_type == "misl"
        target_time = 1100;
    elseif urgency_type == "slow" && trial_type == "easy"
        target_time = 900;
    elseif urgency_type == "slow" && trial_type == "ambi"
        target_time = 1500;
    elseif urgency_type == "slow" && trial_type == "misl"
        target_time = 1500;
    end

    avg_commit_time = mean(commit_time);
    time_diff = abs(avg_commit_time - target_time);
    if time_diff < 200
        score = score - 100;
    else
        temp_credit = -(time_diff - 200) * 0.1 + 100;
        score = score - temp_credit;
    end

    % go through each trial
    for j = 1:num_trials
        % check if the max neuron is located at the right cell
        % interval = 5;
        current_n = max_neuron(1,j);
        target1 = data.exp{1,i}.model.index_neuron1;
        target2 = data.exp{1,i}.model.index_neuron2;
        % condition = ((target1-interval < current_n) && (current_n < target1+interval)) || ((target2-interval < current_n) && (current_n < target2+interval));
        condition = isInRange(current_n,target1) || isInRange(current_n,target2);
        if condition
            score = score - credit;
        end
        % % check the decision time within the reasonable range
        % time_diff = abs(commit_time(1,j) - target_time);
        % if time_diff < 200
        %     score = score - 5;
        % end
    end
end
disp(score);
end
