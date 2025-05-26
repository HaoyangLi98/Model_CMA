function scores = score_function_parallel(X)
    % X: matrix of size [n_params x n_individuals]
    n_individuals = size(X, 2);
    start_t = tic;
    scores = zeros(1, n_individuals);

    parfor i = 1:n_individuals
        output = run_simulation(X(:, i));
        scores(i) = cal_score(output);
    end
    duration = toc(start_t);
    fprintf('>> Total simulation time: %.2f seconds for %d samples\n', duration, n_individuals);
end


