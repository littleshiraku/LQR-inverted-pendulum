%% stage1_openloop
% Stage 1: open-loop unstable response.

clear; clc; close all;
params = common_params();
[A, B] = linear_matrices(params); %#ok<ASGLU>

x0 = params.x0;
openController = @(~, ~) 0;
openResult = simulate_pendulum(params, openController, x0, 2.0, 'Stage 1 open loop');
openMetrics = compute_metrics(openResult, params);
openLoopPoles = eig(A);

figPath = fullfile(params.figDir, 'stage1_openloop_response.png');
fig = figure('Color', 'w', 'Name', 'Stage 1 open-loop response', 'Position', [120, 120, 1000, 680]);
names = {'x (m)', 'x_dot (m/s)', 'theta (rad)', 'theta_dot (rad/s)'};
tiledlayout(4, 1, 'Padding', 'compact', 'TileSpacing', 'compact');
for i = 1:4
    nexttile;
    plot(openResult.t, openResult.x(:, i), 'LineWidth', 1.2, 'Color', [0.75, 0.25, 0.20]);
    grid on;
    ylabel(names{i});
    if i == 1
        title('Stage 1 open-loop unstable response');
    end
end
xlabel('Time (s)');
exportgraphics(fig, figPath, 'Resolution', 160);

animPath = fullfile(params.animDir, 'stage1_openloop.gif');
animate_pendulum(openResult, params, animPath, 30, 2.0);

save(fullfile(params.dataDir, 'stage1_openloop.mat'), ...
    'openResult', 'openMetrics', 'openLoopPoles', 'A', 'B', 'params');

logPath = fullfile(params.logDir, 'stage1_openloop_log.txt');
fid = fopen(logPath, 'w', 'n', 'UTF-8');
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, 'Stage 1 open-loop verification\n');
fprintf(fid, 'Open-loop poles:\n');
fprintf(fid, '% .8f%+ .8fi\n', [real(openLoopPoles), imag(openLoopPoles)]');
fprintf(fid, 'Failure: %s, failure time: %.4f s\n', string(openMetrics.failed), openMetrics.failureTime);
fprintf(fid, 'Cost J: %.8f, Reward: %.8f\n', openMetrics.costJ, openMetrics.reward);

body = sprintf([ ...
    '## 实验目的\n验证直立平衡点开环不稳定性，记录摆杆在短时间内偏离直立的过程。\n\n', ...
    '## 关键参数\n- 初始状态：`[0, 0, 0.1, 0]^T`\n- 控制律：`K=0`, `u=0 N`\n- 仿真时长：`2 s`, 步长：`%.3f s`\n- 开环极点：`%s`\n\n', ...
    '## 结果截图/动画\n![stage1](figures/stage1_openloop_response.png)\n\n', ...
    '动画：[`animations/stage1_openloop.gif`](animations/stage1_openloop.gif)\n\n', ...
    '## 性能指标\n%s\n\n', ...
    '## 结论与不足\n开环矩阵存在右半平面极点，系统自然不稳定；该阶段不施加控制，因此只能作为后续闭环控制的基准。\n'], ...
    params.dt, mat2str(openLoopPoles, 4), metrics_to_markdown(openMetrics));
write_stage_report('阶段 1：开环验证', body, params);

disp('Stage 1 complete.');
