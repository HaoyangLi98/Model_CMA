function output = isInRange(current_idx, target_idx)
    % activity: 1x90 neural activity array
    % target_idx: target neuron index (1~90)
    % returns: true if max neuron is within ±10 neurons of target (circular)

    total_neurons = 90;

    % Compute circular ±10 range
    offset = -10:10;
    range = mod(target_idx + offset - 1, total_neurons) + 1;

    % Check if max neuron index is within range
    output = ismember(current_idx, range);
end
