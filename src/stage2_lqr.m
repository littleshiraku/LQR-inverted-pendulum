%% stage2_lqr
% 阶段 2：基准 LQR 控制器设计与非线性闭环仿真。
%
% 本脚本在直立平衡点附近的线性化模型上设计 LQR 增益 K，
% 再将控制律 u = -Kx 作用到非线性倒立摆模型中，
% 与开环响应对比，验证小扰动条件下的闭环稳定效果。

%% 1. 环境初始化与公共参数读取
% 清空工作区、命令行和图窗，保证每次运行从干净状态开始。
clear; clc; close all;

% common_params() 提供物理参数、LQR 权重、仿真时间、初始状态和输出目录。
params = common_params();

% 获取直立平衡点附近的线性化状态空间矩阵：
%   x_dot = A*x + B*u
% LQR 控制器基于该线性模型求解。
[A, B] = linear_matrices(params);

%% 2. LQR 权重设置与反馈增益求解
% Q 惩罚状态偏差，R 惩罚控制输入幅值。
% 权重越大，优化器越倾向于压低对应状态或控制量。
Q = params.Q_base;
R = params.R_base;

% 求解连续时间 LQR 增益 K，使控制律 u = -K*x 最小化二次型性能指标。
K = lqr(A, B, Q, R);

%% 3. 开环/闭环极点分析
% 闭环矩阵为 A - B*K；若所有闭环极点实部均小于 0，线性闭环系统渐近稳定。
closedLoopPoles = eig(A - B*K);

% 开环极点用于和闭环极点对比，显示 LQR 对系统模态的稳定化作用。
openLoopPoles = eig(A);

%% 4. 开环与 LQR 闭环非线性仿真
% 初始状态与阶段 1 保持一致，便于横向比较开环发散和闭环收敛。
x0 = params.x0;

% 开环基准：控制输入恒为 0。
openResult = simulate_pendulum(params, @(~, ~) 0, x0, params.T, 'Open loop');

% LQR 闭环：当前状态乘以反馈增益并取负号，得到控制力。
% 这里直接在非线性模型中验证线性控制器的实际效果。
closedResult = simulate_pendulum(params, @(~, x) -K*x, x0, params.T, 'LQR closed loop');

%% 5. 性能指标计算
% 使用统一指标评估开环和闭环，包括调节时间、峰值控制力、代价函数等。
openMetrics = compute_metrics(openResult, params);
closedMetrics = compute_metrics(closedResult, params);

%% 6. 绘制开环与闭环对比曲线
% plot_results 将两组仿真结果画在同一张图中，突出 LQR 对状态发散的抑制效果。
figPath = fullfile(params.figDir, 'stage2_lqr_vs_openloop.png');
plot_results(openResult, closedResult, figPath, 'Stage 2 LQR closed-loop stabilization');

%% 7. 生成闭环动画
% 只对 LQR 闭环结果生成动画，用于观察小车和摆杆如何回到平衡附近。
animPath = fullfile(params.animDir, 'stage2_lqr.gif');
animate_pendulum(closedResult, params, animPath, 30, 8.0);

%% 8. 保存实验数据
% 保存控制器、极点、仿真结果和指标，便于后续调参、画图或报告复现。
save(fullfile(params.dataDir, 'stage2_lqr.mat'), ...
    'A', 'B', 'Q', 'R', 'K', 'openLoopPoles', 'closedLoopPoles', ...
    'openResult', 'closedResult', 'openMetrics', 'closedMetrics', 'params');

%% 9. 写入文本日志
% 日志输出关键数值，方便快速检查 LQR 增益、极点稳定性和主要性能指标。
logPath = fullfile(params.logDir, 'stage2_lqr_log.txt');
fid = fopen(logPath, 'w', 'n', 'UTF-8');

% onCleanup 确保文件句柄在脚本结束或异常退出时被关闭。
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

%% 10. 生成阶段 Markdown 报告
% 报告汇总实验目标、控制器参数、极点、曲线、动画和性能指标。
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

%% 11. 命令行完成提示
disp('Stage 2 complete.');
