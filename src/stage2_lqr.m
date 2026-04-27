%% stage2_lqr
% Stage 2: baseline LQR design and nonlinear closed-loop simulation.

clear; clc; close all;
params = common_params();
[A, B] = linear_matrices(params);

Q = params.Q_base;
R = params.R_base;
K = lqr(A, B, Q, R);
closedLoopPoles = eig(A - B*K);
openLoopPoles = eig(A);

x0 = params.x0;
openResult = simulate_pendulum(params, @(~, ~) 0, x0, params.T, 'Open loop');
closedResult = simulate_pendulum(params, @(~, x) -K*x, x0, params.T, 'LQR closed loop');
openMetrics = compute_metrics(openResult, params);
closedMetrics = compute_metrics(closedResult, params);

figPath = fullfile(params.figDir, 'stage2_lqr_vs_openloop.png');
plot_results(openResult, closedResult, figPath, 'Stage 2 LQR closed-loop stabilization');

animPath = fullfile(params.animDir, 'stage2_lqr.gif');
animate_pendulum(closedResult, params, animPath, 30, 8.0);

save(fullfile(params.dataDir, 'stage2_lqr.mat'), ...
    'A', 'B', 'Q', 'R', 'K', 'openLoopPoles', 'closedLoopPoles', ...
    'openResult', 'closedResult', 'openMetrics', 'closedMetrics', 'params');

logPath = fullfile(params.logDir, 'stage2_lqr_log.txt');
fid = fopen(logPath, 'w', 'n', 'UTF-8');
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, 'Stage 2 LQR baseline\n');
fprintf(fid, 'Q = diag([15 5 180 50]), R = 0.15\n');
fprintf(fid, 'K:\n');
fprintf(fid, '% .8f ', K);
fprintf(fid, '\nOpen-loop poles:\n');
fprintf(fid, '% .8f%+ .8fi\n', [real(openLoopPoles), imag(openLoopPoles)]');
fprintf(fid, 'Closed-loop poles:\n');
fprintf(fid, '% .8f%+ .8fi\n', [real(closedLoopPoles), imag(closedLoopPoles)]');
fprintf(fid, 'Closed-loop all stable: %s\n', string(all(real(closedLoopPoles) < 0)));
fprintf(fid, 'Theta settling time: %.6g\n', closedMetrics.thetaSettlingTime);
fprintf(fid, 'X settling time: %.6g\n', closedMetrics.xSettlingTime);
fprintf(fid, 'U peak: %.6g\n', closedMetrics.uPeak);
fprintf(fid, 'Cost J: %.8f\n', closedMetrics.costJ);

body = sprintf([ ...
    '## 实验目的\n设计 LQR 控制器并验证非线性倒立摆可在小扰动下稳定到直立平衡点。\n\n', ...
    '## 关键参数\n- `Q=diag([15, 5, 180, 50])`, `R=0.15`\n- `K=%s`\n- 开环极点：`%s`\n- 闭环极点：`%s`\n- 闭环极点全部位于左半平面：`%s`\n\n', ...
    '## 结果截图/动画\n![stage2](figures/stage2_lqr_vs_openloop.png)\n\n', ...
    '动画：[`animations/stage2_lqr.gif`](animations/stage2_lqr.gif)\n\n', ...
    '## 性能指标\n%s\n\n', ...
    '## 结论与不足\nLQR 可显著抑制开环发散，角度和位置均回到指定误差带；该控制器基于小扰动线性化，大角度初值下仍需额外保护或摆起控制。\n'], ...
    mat2str(K, 5), mat2str(openLoopPoles, 4), mat2str(closedLoopPoles, 4), ...
    string(all(real(closedLoopPoles) < 0)), metrics_to_markdown(closedMetrics));
write_stage_report('阶段 2：LQR 闭环', body, params);

disp('Stage 2 complete.');
