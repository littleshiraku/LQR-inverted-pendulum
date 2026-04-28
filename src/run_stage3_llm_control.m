%% run_stage3_llm_control
% 通过外部 Python 进程启动阶段 3，兼容 MATLAB R2023a。
%
% 阶段 3 的核心控制器在 Python 中实现，因为 LoRA/Unsloth/Transformers
% 推理环境更适合由 Python 管理。本脚本负责：
%   1. 切换到项目根目录并配置路径；
%   2. 调用 stage3_llm_control.py 生成结果文件；
%   3. 读取 Python 输出的 MAT 文件，在 MATLAB 中复绘曲线和动画。

%% 1. MATLAB 环境初始化
% 清理命令行和变量，但不关闭图窗，便于用户保留已有对比图。
clear; clc;

% 定位当前脚本、src 目录和项目根目录。
thisFile = mfilename('fullpath');
srcDir = fileparts(thisFile);
rootDir = fileparts(srcDir);

% 临时切换到项目根目录，脚本结束或异常时自动恢复原工作目录。
oldDir = pwd;
cleanup = onCleanup(@() cd(oldDir));
cd(rootDir);

% 将 src 加入 MATLAB 路径，以便调用 common_params、animate_pendulum 等项目函数。
addpath(srcDir);

% 强制 Python 子进程使用 UTF-8 输出，避免中文日志在 Windows 终端中乱码。
setenv('PYTHONIOENCODING', 'utf-8');

%% 2. 选择 Python 解释器
% 可通过环境变量 STAGE3_PYTHON 指定专用虚拟环境；未指定时使用系统 python。
pythonExe = getenv('STAGE3_PYTHON');
if isempty(pythonExe)
    pythonExe = 'python';
end

%% 3. 启动阶段 3 Python 主脚本
% Python 脚本会负责加载 LoRA 控制器、执行仿真并写出 data/ 下的结果文件。
stage3Script = fullfile(srcDir, 'stage3_llm_control.py');
cmd = sprintf('"%s" "%s"', pythonExe, stage3Script);

fprintf('Running Stage 3 LoRA Llama direct control with:\n%s\n\n', cmd);
[status, output] = system(cmd);
fprintf('%s\n', output);

% 若 Python 进程返回非零退出码，直接抛出 MATLAB 错误并停止后续绘图。
if status ~= 0
    error('Stage 3 LoRA Llama direct control failed with exit code %d.', status);
end

%% 4. 读取并展示阶段 3 结果
% Python 已生成 MAT 文件；这里复用 MATLAB 侧绘图和动画工具展示结果。
show_stage3_results(rootDir);
disp('Stage 3 LoRA Llama direct control complete.');

function show_stage3_results(rootDir)
%SHOW_STAGE3_RESULTS 读取 Python 结果文件，并生成 MATLAB 侧图像/动画。
%
% 输入:
%   rootDir - 项目根目录，用于定位 data 和 animations 输出目录。

% 读取公共参数，主要供 animate_pendulum 获取摆长、绘图范围等设置。
params = common_params();
matPath = fullfile(rootDir, 'data', 'stage3_llm_control.mat');
if ~exist(matPath, 'file')
    error('Stage 3 result file was not found: %s', matPath);
end

data = load(matPath);

% 将 MAT 文件中的数组包装成与项目其他 MATLAB 仿真函数一致的 result 结构。
llmResult = make_result(data.t, data.x_llm, data.u_llm, 'LoRA Llama direct');

% 如果 Python 环境安装了 SciPy，会同时保存 LQR 基准轨迹；否则只展示 LLM 结果。
hasLqr = isfield(data, 'x_lqr') && isfield(data, 'u_lqr');
if hasLqr
    lqrResult = make_result(data.t, data.x_lqr, data.u_lqr, 'LQR baseline');
else
    lqrResult = [];
end

plot_stage3_results(llmResult, lqrResult);

% 使用 MATLAB 项目内统一动画函数重新生成 GIF，保证风格与前两阶段一致。
animPath = fullfile(rootDir, 'animations', 'stage3_llm_control.gif');
animate_pendulum(llmResult, params, animPath, 30, 8.0);
end

function result = make_result(t, x, u, label)
%MAKE_RESULT 将时间、状态、控制量数组封装为统一 result 结构。
result = struct();
result.t = t(:);
result.x = x;
result.u = u(:);
result.label = label;
end

function plot_stage3_results(llmResult, lqrResult)
%PLOT_STAGE3_RESULTS 绘制阶段 3 的状态轨迹和控制输入。
%
% 前四个子图对应状态向量 [x, x_dot, theta, theta_dot]，
% 第五个子图对应控制力 u。若存在 LQR 基准，则与 LLM 结果叠加对比。
stateNames = {'x (m)', 'x_dot (m/s)', 'theta (rad)', 'theta_dot (rad/s)'};
figure('Color', 'w', 'Name', 'Stage 3 LoRA Llama direct control', ...
    'Position', [100, 100, 1100, 780]);
tiledlayout(5, 1, 'Padding', 'compact', 'TileSpacing', 'compact');

% 绘制四个状态量随时间变化的曲线。
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
% 绘制控制力曲线；颜色与状态图保持一致。
if ~isempty(lqrResult)
    plot(lqrResult.t, lqrResult.u, 'Color', [0.10 0.35 0.75], 'LineWidth', 1.1);
    hold on;
end
plot(llmResult.t, llmResult.u, 'Color', [0.80 0.20 0.18], 'LineWidth', 1.2);
grid on;
ylabel('u (N)');
xlabel('Time (s)');

% 立即刷新图窗，便于从 MATLAB 命令行运行时直接看到结果。
drawnow;
end
