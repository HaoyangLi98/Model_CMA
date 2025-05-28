function score = score_function(x)
    % Simulate the model
    % try
    %     output = run_simulation(x);
    % catch
    %     % If the simulation fails, return a high penalty
    %     score = -1e6;
    %     return;
    % end

    output = run_simulation(x);
    % === Define qualitative scoring ===
    score = cal_score(output);
end
