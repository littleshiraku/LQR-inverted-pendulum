function plot_results(openResult, closedResult, outPath, figTitle)
%PLOT_RESULTS Overlay open-loop and closed-loop state/control histories.

if nargin < 4
    figTitle = 'Cart-pole response';
end

stateNames = {'x (m)', 'x_dot (m/s)', 'theta (rad)', 'theta_dot (rad/s)'};
f = figure('Color', 'w', 'Name', figTitle, 'Position', [100, 100, 1100, 780]);
tiledlayout(5, 1, 'Padding', 'compact', 'TileSpacing', 'compact');

for i = 1:4
    nexttile;
    plot(openResult.t, openResult.x(:, i), 'Color', [0.75 0.25 0.20], 'LineWidth', 1.0);
    hold on;
    plot(closedResult.t, closedResult.x(:, i), 'Color', [0.10 0.35 0.75], 'LineWidth', 1.2);
    grid on;
    ylabel(stateNames{i});
    if i == 1
        title(figTitle, 'Interpreter', 'none');
        legend({'Open loop', 'Controlled'}, 'Location', 'best');
    end
end

nexttile;
plot(openResult.t, openResult.u, 'Color', [0.75 0.25 0.20], 'LineWidth', 1.0);
hold on;
plot(closedResult.t, closedResult.u, 'Color', [0.10 0.35 0.75], 'LineWidth', 1.2);
grid on;
ylabel('u (N)');
xlabel('Time (s)');

if nargin >= 3 && ~isempty(outPath)
    exportgraphics(f, outPath, 'Resolution', 160);
end
end
