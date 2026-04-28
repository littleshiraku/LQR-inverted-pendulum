%% run_stage3_llm_control
% Launch Stage 3 through an external Python process for MATLAB R2023a.

clear; clc;

thisFile = mfilename('fullpath');
srcDir = fileparts(thisFile);
rootDir = fileparts(srcDir);
oldDir = pwd;
cleanup = onCleanup(@() cd(oldDir));
cd(rootDir);
addpath(srcDir);
setenv('PYTHONIOENCODING', 'utf-8');

pythonExe = getenv('STAGE3_PYTHON');
if isempty(pythonExe)
    pythonExe = 'python';
end

stage3Script = fullfile(srcDir, 'stage3_llm_control.py');
cmd = sprintf('"%s" "%s"', pythonExe, stage3Script);

fprintf('Running Stage 3 LoRA Llama direct control with:\n%s\n\n', cmd);
[status, output] = system(cmd);
fprintf('%s\n', output);

if status ~= 0
    error('Stage 3 LoRA Llama direct control failed with exit code %d.', status);
end

show_stage3_results(rootDir);
disp('Stage 3 LoRA Llama direct control complete.');

function show_stage3_results(rootDir)
params = common_params();
matPath = fullfile(rootDir, 'data', 'stage3_llm_control.mat');
if ~exist(matPath, 'file')
    error('Stage 3 result file was not found: %s', matPath);
end

data = load(matPath);
llmResult = make_result(data.t, data.x_llm, data.u_llm, 'LoRA Llama direct');

hasLqr = isfield(data, 'x_lqr') && isfield(data, 'u_lqr');
if hasLqr
    lqrResult = make_result(data.t, data.x_lqr, data.u_lqr, 'LQR baseline');
else
    lqrResult = [];
end

plot_stage3_results(llmResult, lqrResult);

animPath = fullfile(rootDir, 'animations', 'stage3_llm_control.gif');
animate_pendulum(llmResult, params, animPath, 30, 8.0);
end

function result = make_result(t, x, u, label)
result = struct();
result.t = t(:);
result.x = x;
result.u = u(:);
result.label = label;
end

function plot_stage3_results(llmResult, lqrResult)
stateNames = {'x (m)', 'x_dot (m/s)', 'theta (rad)', 'theta_dot (rad/s)'};
figure('Color', 'w', 'Name', 'Stage 3 LoRA Llama direct control', ...
    'Position', [100, 100, 1100, 780]);
tiledlayout(5, 1, 'Padding', 'compact', 'TileSpacing', 'compact');

for i = 1:4
    nexttile;
    if ~isempty(lqrResult)
        plot(lqrResult.t, lqrResult.x(:, i), 'Color', [0.10 0.35 0.75], 'LineWidth', 1.1);
        hold on;
    end
    plot(llmResult.t, llmResult.x(:, i), 'Color', [0.80 0.20 0.18], 'LineWidth', 1.2);
    grid on;
    ylabel(stateNames{i});
    if i == 1
        title('Stage 3 LoRA Llama direct control', 'Interpreter', 'none');
        if ~isempty(lqrResult)
            legend({'LQR baseline', 'LoRA Llama direct'}, 'Location', 'best');
        else
            legend({'LoRA Llama direct'}, 'Location', 'best');
        end
    end
end

nexttile;
if ~isempty(lqrResult)
    plot(lqrResult.t, lqrResult.u, 'Color', [0.10 0.35 0.75], 'LineWidth', 1.1);
    hold on;
end
plot(llmResult.t, llmResult.u, 'Color', [0.80 0.20 0.18], 'LineWidth', 1.2);
grid on;
ylabel('u (N)');
xlabel('Time (s)');

drawnow;
end
