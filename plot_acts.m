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
