function metrics = compute_metrics(result, params)
%COMPUTE_METRICS Response metrics used across all stages.

t = result.t(:);
x = result.x(:, 1);
theta = result.x(:, 3);
u = result.u(:);

metrics.label = result.label;
metrics.failed = result.failed;
metrics.failureTime = result.failureTime;
metrics.thetaSettlingTime = settling_time(t, theta, params.theta_band);
metrics.xSettlingTime = settling_time(t, x, params.x_band);
metrics.thetaOvershoot = overshoot_abs(theta);
metrics.xOvershoot = overshoot_abs(x);
metrics.thetaSteadyStateError = abs(mean_tail(theta));
metrics.xSteadyStateError = abs(mean_tail(x));
metrics.uPeak = max(abs(u));
metrics.uEnergy = trapz(t, u.^2);
metrics.costJ = trapz(t, 10*theta.^2 + x.^2 + 0.01*u.^2) + 1000*double(result.failed);
metrics.reward = -metrics.costJ;
end

function ts = settling_time(t, y, band)
idx = find(abs(y) > band, 1, 'last');
if isempty(idx)
    ts = 0;
elseif idx >= numel(t)
    ts = Inf;
else
    ts = t(idx + 1);
end
end

function y = mean_tail(signal)
n = numel(signal);
tailStart = max(1, floor(0.95*n));
y = mean(signal(tailStart:end));
end

function os = overshoot_abs(signal)
initial = abs(signal(1));
peak = max(abs(signal));
os = max(0, peak - initial);
end
