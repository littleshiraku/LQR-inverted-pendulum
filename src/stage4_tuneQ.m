%% stage4_tuneQ
% Stage 4: offline simulated-LLM search over Q and R for LQR tuning.

clear; clc; close all;
params = common_params();
[A, B] = linear_matrices(params);
x0 = params.x0;

baseline.Q = params.Q_base;
baseline.R = params.R_base;
baseline.K = lqr(A, B, baseline.Q, baseline.R);
baseline.result = simulate_pendulum(params, @(~, x) -baseline.K*x, x0, params.T, 'Baseline LQR');
baseline.metrics = compute_metrics(baseline.result, params);

useBayes = exist('bayesopt', 'file') == 2 && exist('optimizableVariable', 'file') == 2;
history = [];

if useBayes
    rng(42);
    vars = [
        optimizableVariable('q1', [1, 20])
        optimizableVariable('q2', [1, 10])
        optimizableVariable('q3', [50, 200])
        optimizableVariable('q4', [5, 50])
        optimizableVariable('r', [0.1, 5])
    ];
    objective = @(tbl) tune_objective([tbl.q1, tbl.q2, tbl.q3, tbl.q4, tbl.r], A, B, params, x0);
    bo = bayesopt(objective, vars, ...
        'MaxObjectiveEvaluations', 30, ...
        'IsObjectiveDeterministic', true, ...
        'AcquisitionFunctionName', 'expected-improvement-plus', ...
        'Verbose', 0, ...
        'PlotFcn', {});
    bestX = bo.XAtMinObjective;
    bestVec = [bestX.q1, bestX.q2, bestX.q3, bestX.q4, bestX.r];
    history = table2array(bo.XTrace);
    history(:, end+1) = bo.ObjectiveTrace;
    searchMethod = 'bayesopt';
else
    [bestVec, history] = fallback_llm_search(A, B, params, x0);
    searchMethod = 'offline deterministic simulated-LLM search';
end

Qstar = diag(bestVec(1:4));
Rstar = bestVec(5);
Kstar = lqr(A, B, Qstar, Rstar);
bestResult = simulate_pendulum(params, @(~, x) -Kstar*x, x0, params.T, 'Tuned LQR');
bestMetrics = compute_metrics(bestResult, params);
closedLoopPoles = eig(A - B*Kstar);

improvement.cost = percent_improve(baseline.metrics.costJ, bestMetrics.costJ);
improvement.uEnergy = percent_improve(baseline.metrics.uEnergy, bestMetrics.uEnergy);
improvement.thetaSettling = percent_improve(baseline.metrics.thetaSettlingTime, bestMetrics.thetaSettlingTime);
improvement.xSettling = percent_improve(baseline.metrics.xSettlingTime, bestMetrics.xSettlingTime);

responseFig = fullfile(params.figDir, 'stage4_tuned_vs_baseline.png');
plot_results(baseline.result, bestResult, responseFig, 'Stage 4 tuned Q/R versus baseline LQR');

searchFig = fullfile(params.figDir, 'stage4_search_curve.png');
plot_search_history(history, searchFig, searchMethod);

animPath = fullfile(params.animDir, 'stage4_tuned_lqr.gif');
animate_pendulum(bestResult, params, animPath, 30, 8.0);

save(fullfile(params.dataDir, 'stage4_tuneQ.mat'), ...
    'baseline', 'bestVec', 'Qstar', 'Rstar', 'Kstar', 'bestResult', ...
    'bestMetrics', 'closedLoopPoles', 'history', 'improvement', 'searchMethod', 'params');

logPath = fullfile(params.logDir, 'stage4_tuneQ_log.txt');
fid = fopen(logPath, 'w', 'n', 'UTF-8');
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, 'Stage 4 offline simulated-LLM Q/R tuning\n');
fprintf(fid, 'Search method: %s\n', searchMethod);
fprintf(fid, 'Best vector [q1 q2 q3 q4 R]: %s\n', mat2str(bestVec, 8));
fprintf(fid, 'Kstar: %s\n', mat2str(Kstar, 8));
fprintf(fid, 'Closed-loop poles: %s\n', mat2str(closedLoopPoles, 8));
fprintf(fid, 'Baseline J: %.8f, Tuned J: %.8f, improvement: %.4f%%\n', ...
    baseline.metrics.costJ, bestMetrics.costJ, improvement.cost);

body = sprintf([ ...
    '## 实验目的\n用离线模拟 LLM 推荐器搜索 `Q` 对角元素和 `R`，比较自动调参 LQR 与传统试调 LQR 的指标差异。\n\n', ...
    '## 关键参数\n- 搜索空间：`q1∈[1,20]`, `q2∈[1,10]`, `q3∈[50,200]`, `q4∈[5,50]`, `R∈[0.1,5]`\n', ...
    '- 搜索方法：`%s`\n- 最优 `Q*=%s`, `R*=%.5g`\n- 最优 `K*=%s`\n- 闭环极点：`%s`\n\n', ...
    '## 结果截图/动画\n![stage4-search](figures/stage4_search_curve.png)\n\n', ...
    '![stage4-response](figures/stage4_tuned_vs_baseline.png)\n\n', ...
    '动画：[`animations/stage4_tuned_lqr.gif`](animations/stage4_tuned_lqr.gif)\n\n', ...
    '## 性能指标\n%s\n\n', ...
    '## 相对传统试调法提升\n- 代价 J 降低：`%.2f%%`\n- 控制能量降低：`%.2f%%`\n- theta 调节时间缩短：`%.2f%%`\n- x 调节时间缩短：`%.2f%%`\n\n', ...
    '## 结论与不足\n离线推荐搜索可在给定代价函数下改善基准 LQR；该阶段没有接入真实 qwen API，因此结果代表可复现实验框架和启发式推荐能力，不代表真实大模型在线控制性能。\n'], ...
    searchMethod, mat2str(diag(Qstar)', 5), Rstar, mat2str(Kstar, 5), ...
    mat2str(closedLoopPoles, 4), metrics_to_markdown(bestMetrics), ...
    improvement.cost, improvement.uEnergy, improvement.thetaSettling, improvement.xSettling);
write_stage_report('阶段 4：LLM 调 Q 矩阵', body, params);

disp('Stage 4 complete.');

function cost = tune_objective(vec, A, B, params, x0)
Q = diag(vec(1:4));
R = vec(5);
try
    K = lqr(A, B, Q, R);
    result = simulate_pendulum(params, @(~, x) -K*x, x0, params.T, 'candidate');
    metrics = compute_metrics(result, params);
    cost = metrics.costJ;
catch
    cost = 1e9;
end
end

function [bestVec, history] = fallback_llm_search(A, B, params, x0)
rng(42);
seed = [
    15, 5, 180, 50, 0.15
    10, 1, 150, 20, 1.0
    8,  2, 180, 30, 0.8
    5,  1, 200, 45, 0.5
    12, 4, 160, 25, 1.5
    3,  2, 120, 35, 0.4
];
nRandom = 35;
lo = [1, 1, 50, 5, 0.1];
hi = [20, 10, 200, 50, 5];
samples = [seed; lo + rand(nRandom, 5).*(hi-lo)];

history = zeros(size(samples, 1), 6);
bestCost = Inf;
bestVec = samples(1, :);
for i = 1:size(samples, 1)
    vec = samples(i, :);
    cost = tune_objective(vec, A, B, params, x0);
    history(i, :) = [vec, cost];
    if cost < bestCost
        bestCost = cost;
        bestVec = vec;
    end
end
end

function plot_search_history(history, outPath, searchMethod)
fig = figure('Color', 'w', 'Name', 'Stage 4 search curve', 'Position', [120, 120, 900, 480]);
cost = history(:, end);
bestSoFar = cummin(cost);
plot(1:numel(cost), cost, 'o-', 'Color', [0.65, 0.65, 0.65], 'LineWidth', 1.0);
hold on;
plot(1:numel(cost), bestSoFar, 'LineWidth', 2.0, 'Color', [0.10, 0.35, 0.75]);
grid on;
xlabel('Iteration');
ylabel('Cost J');
title(['Stage 4 search curve: ', searchMethod], 'Interpreter', 'none');
legend({'Candidate cost', 'Best so far'}, 'Location', 'best');
exportgraphics(fig, outPath, 'Resolution', 160);
end

function pct = percent_improve(baseValue, newValue)
if ~isfinite(baseValue) || baseValue == 0
    pct = NaN;
else
    pct = 100 * (baseValue - newValue) / abs(baseValue);
end
end
