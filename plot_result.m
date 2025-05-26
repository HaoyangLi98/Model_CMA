function plot_result(data,plot_acts,plot_behaviour)
% data: 1x6 cell array (6 conditions), each cell has 1x20 trials
% each trial.neural_acts.area = [n_neurons x time]
% each trial.decision_time = scalar
n_conditions = 6;
time = data.exp{1,1}.model.timeSteps;
if plot_acts == 1
for cond = 1:n_conditions
    [left,right] = avg_trials(data.acts{1,cond},data.exp{1,cond}.model);
    figure;
    plot_neural_activity_3d(left, time);
end
end

if plot_behaviour == 1
   
    plot_decisionTime(data);
    plot_choices(data);
    plot_both(data);
end

end


%% helper functions
function plot_both(data)
    blocks = {'slow', 'fast'};
    conds = {'easy', 'ambiguous', 'misleading'};
    colors = [0 0 1; 0 0.6 0; 1 0 0];  % blue, green, red
    edges = 0:200:3000;  % bin edges in seconds
    centers = edges(1:end-1) + diff(edges)/2;

    figure;
    cond = 0;
    for b = 1:2
        subplot(2,1,b);
        for c = 1:3
            cond = cond + 1;
            dts = [data.acts{1,cond}.commit_time];
            [~,correct] = cal_choice(data.acts{1,cond},data.exp{1,cond}.model);

            % Histogram
            h_total = histcounts(dts, edges, 'Normalization', 'probability') * 100;
            h_err = histcounts(dts(correct == 0), edges) / numel(dts) * 100;
            hold on;
            % Shaded error area
            area(centers, h_err, 'FaceColor', colors(c,:), 'FaceAlpha', 0.2, 'EdgeAlpha', 0,'HandleVisibility','off');
            hold on;
            % Line plot
            plot(centers, h_total, 'Color', colors(c,:), 'LineWidth', 1.5,'DisplayName', conds{c});
            hold on;
            % Mean decision time
            mu = mean(dts);
            xline(mu, '--', 'Color', colors(c,:), 'LineWidth', 1,'HandleVisibility','off');
            text(mu, max(h_total) * .8, sprintf('%d', round(mu)), ...
                'Color', colors(c,:), 'FontSize', 10, 'FontWeight', 'bold', ...
                'HorizontalAlignment', 'center');

            if c == 3
                legend('Location', 'northeast');  % use labels from DisplayName
            end
        end
        xlabel('Decision Time (ms)');
        ylabel('% of Trials');
        title(data.exp{1,cond}.urgency);
        xlim([0 3000]);
        ylim([0 100]);

    end
end


function plot_choices(data)

    n_conditions = 6;
    classNames = {'incorrect', 'correct', 'wrong'};
    figure;
    for cond = 1:n_conditions
        
        [counts,~] = cal_choice(data.acts{1,cond},data.exp{1,cond}.model);
        urgency = data.exp{1,cond}.urgency;
        trial_type = data.exp{1,cond}.trial_type;
        name = [urgency,'-', trial_type];

        % Plot histogram (bar chart)
        subplot(2,3,cond);
        bar(counts, 'FaceColor', 'b', 'EdgeColor', 'k');
        
        % Customize x-axis labels
        xticklabels(classNames); % Apply custom class names
        
        % Labels and title
        xlabel('Value');
        ylabel('Count');
        title(name);
        grid on;
        hold on;
    end
end


function plot_decisionTime(data)
    n_conditions = 6;
    x_axis = [];
    for cond = 1:n_conditions
        % cal for decision times
        dts = [data.acts{1,cond}.commit_time];
        mean_dt(cond) = mean(dts);
        std_dt(cond) = std(dts);
        urgency = data.exp{1,cond}.urgency;
        trial_type = data.exp{1,cond}.trial_type;
        name = string([urgency,'-', trial_type]);
        x_axis = [x_axis,name];
    end
    % plot decision time
    figure;
    % x_axis = ["easy&fast","ambi&fast","misl&fast","easy&slow","ambi&slow","misl&slow"];
    bar(x_axis,mean_dt);
    hold on;
    errorbar(1:n_conditions, mean_dt, std_dt, 'k.', 'LineWidth', 1);
    
    for i = 1:n_conditions
        text(i, mean_dt(i) + std_dt(i) + 0.02, ...
             sprintf('%.2f', mean_dt(i)), ...
             'HorizontalAlignment', 'center', ...
             'VerticalAlignment', 'bottom', ...
             'FontSize', 10, ...
             'FontWeight', 'bold');
    end
    xlabel('Condition');
    ylabel('Decision Time');
    title('Mean Decision Time ± STD');
end


function plot_neural_activity_3d(avg_activity, time)
% avg_activity: struct with region fields (e.g., .PFC), each [neurons x time]
% time: 1xT time vector

    regions = fieldnames(avg_activity);
    n_regions = numel(regions);
    cutting_point = find(time == -300);

    for i = 1:n_regions
        region = regions{i};
        A = avg_activity.(region);  % [neurons x time]
        [n_neurons, ~] = size(A);

        % Create meshgrid
        [T_grid, N_grid] = meshgrid(time, 1:n_neurons);  % [neurons x time]
        [~,T] = size(T_grid);
        % Plot 3D surface
        subplot(2,2,i);
        surf(T_grid(:,cutting_point:end), N_grid(:,cutting_point:end), A(:,cutting_point:T), 'EdgeColor', 'none');
        colormap turbo;
        colorbar;

        xlabel('Time');
        ylabel('Neuron Index');
        zlabel('Activity');
        title([region ' Neural Activity (3D Surface)']);
        view(45, 30);  % set a nice 3D view angle
        shading interp;
    end
end


