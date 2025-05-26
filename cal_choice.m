function [counts, correct_list]= cal_choice(acts,model)
    data = [acts.max_neuron];
    template = [acts.direction];
    
    trial_num = length(acts);
    % Count occurrences of each class
    correct = 0;
    incorrect = 0;
    wrong = 0;
    correct_list = zeros(1,trial_num);
    for i = 1:trial_num
        temp = data(i);
        if template(i) == 1
            correct_choice = model.index_neuron2;
            incorrect_choice = model.index_neuron1;
        else
            correct_choice = model.index_neuron1;
            incorrect_choice = model.index_neuron2;
        end
    
        if isInRange(temp,correct_choice) %temp == correct_choice
            correct = correct+1;
            correct_list(1,i) = 1;
        elseif isInRange(temp,incorrect_choice) %temp == incorrect_choice
            incorrect = incorrect+1;
        else
            wrong = wrong+1;
        end
    end
    counts = [incorrect, correct, wrong];
end
