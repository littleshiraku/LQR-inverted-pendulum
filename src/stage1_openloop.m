%% stage1_openloop
% 阶段 1：开环不稳定响应验证。
%
% 本脚本用于验证倒立摆在直立平衡点附近的小扰动开环响应。
% 控制输入固定为 0，因此系统只受自身非线性动力学影响；
% 若线性化系统存在右半平面极点，摆杆角度会随时间发散。

%% 1. 环境初始化与公共参数读取
% 清空工作区、命令行和图窗，避免上一次运行残留变量或图像影响本次实验。
clear; clc; close all;

% common_params() 统一管理物理参数、仿真步长、初始状态和输出目录。
params = common_params();

% 获取直立平衡点附近的线性化状态空间矩阵。
% 本阶段只需要 A 矩阵计算开环极点，B 矩阵随结果一起保存，便于后续复核。
[A, B] = linear_matrices(params);

%% 2. 开环控制器定义与非线性仿真
% 初始状态 x = [小车位置, 小车速度, 摆杆角度, 摆杆角速度]^T。
% params.x0 中默认给出一个小角度扰动，用于观察直立点附近的自然发散。
x0 = params.x0;

% 开环控制律：任意时刻、任意状态下都不施加控制力。
% simulate_pendulum 期望控制器形式为 controller(t, x)，因此这里保留两个入参。
openController = @(~, ~) 0;

% 进行 2 s 非线性仿真。阶段 1 只需要短时间窗口即可观察不稳定趋势。
openResult = simulate_pendulum(params, openController, x0, 2.0, 'Stage 1 open loop');

%% 3. 指标与极点分析
% 计算失败标志、失败时间、积分代价、奖励等统一性能指标。
openMetrics = compute_metrics(openResult, params);

% 线性化开环系统极点；若存在实部大于 0 的极点，说明直立平衡点开环不稳定。
openLoopPoles = eig(A);

%% 4. 绘制状态响应曲线
% 将四个状态量分成四行绘制，便于直接观察角度和位置是否发散。
figPath = fullfile(params.figDir, 'stage1_openloop_response.png');
fig = figure('Color', 'w', 'Name', 'Stage 1 open-loop response', 'Position', [120, 120, 1000, 680]);
names = {'x (m)', 'x_dot (m/s)', 'theta (rad)', 'theta_dot (rad/s)'};
tiledlayout(4, 1, 'Padding', 'compact', 'TileSpacing', 'compact');
for i = 1:4
    nexttile;

    % openResult.x 的第 i 列对应 names{i} 中标注的状态变量。
    plot(openResult.t, openResult.x(:, i), 'LineWidth', 1.2, 'Color', [0.75, 0.25, 0.20]);
    grid on;
    ylabel(names{i});

    % 只在第一幅子图上放标题，避免每个子图重复标题造成拥挤。
    if i == 1
        title('Stage 1 open-loop unstable response');
    end
end
xlabel('Time (s)');

% 导出 PNG 图片，用于阶段报告中的结果截图。
exportgraphics(fig, figPath, 'Resolution', 160);

%% 5. 生成摆杆动画
% 将开环仿真过程导出为 GIF，直观展示摆杆偏离直立位置的过程。
animPath = fullfile(params.animDir, 'stage1_openloop.gif');
animate_pendulum(openResult, params, animPath, 30, 2.0);

%% 6. 保存实验数据
% 保存原始仿真结果、指标、线性化矩阵和参数，便于后续脚本或人工复现实验。
save(fullfile(params.dataDir, 'stage1_openloop.mat'), ...
    'openResult', 'openMetrics', 'openLoopPoles', 'A', 'B', 'params');

%% 7. 写入文本日志
% 日志记录关键数值结果，便于不打开 MAT 文件时快速检查实验是否符合预期。
logPath = fullfile(params.logDir, 'stage1_openloop_log.txt');
fid = fopen(logPath, 'w', 'n', 'UTF-8');

% onCleanup 保证脚本正常结束或中途报错时都能关闭文件句柄。
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, 'Stage 1 open-loop verification\n');
fprintf(fid, 'Open-loop poles:\n');
fprintf(fid, '% .8f%+ .8fi\n', [real(openLoopPoles), imag(openLoopPoles)]');
fprintf(fid, 'Failure: %s, failure time: %.4f s\n', string(openMetrics.failed), openMetrics.failureTime);
fprintf(fid, 'Cost J: %.8f, Reward: %.8f\n', openMetrics.costJ, openMetrics.reward);

%% 8. 生成阶段 Markdown 报告
% 报告正文会被 write_stage_report 写入指定报告目录，图片和动画路径使用相对路径引用。
body = sprintf([ ...
    '## 实验目的\n验证直立平衡点开环不稳定性，记录摆杆在短时间内偏离直立的过程。\n\n', ...
    '## 关键参数\n- 初始状态：`[0, 0, 0.1, 0]^T`\n- 控制律：`K=0`, `u=0 N`\n- 仿真时长：`2 s`, 步长：`%.3f s`\n- 开环极点：`%s`\n\n', ...
    '## 结果截图/动画\n![stage1](figures/stage1_openloop_response.png)\n\n', ...
    '动画：[`animations/stage1_openloop.gif`](animations/stage1_openloop.gif)\n\n', ...
    '## 性能指标\n%s\n\n', ...
    '## 结论与不足\n开环矩阵存在右半平面极点，系统自然不稳定；该阶段不施加控制，因此只能作为后续闭环控制的基准。\n'], ...
    params.dt, mat2str(openLoopPoles, 4), metrics_to_markdown(openMetrics));
write_stage_report('阶段 1：开环验证', body, params);

%% 9. 命令行完成提示
disp('Stage 1 complete.');
