function result = simulate_pendulum(params, controller, x0, T, label)
%SIMULATE_PENDULUM Fixed-step nonlinear simulation with zero-order hold.
%
% controller is a function handle: u = controller(t, state).

if nargin < 5
    label = "simulation";
end

dt = params.dt;
t = (0:dt:T)';
n = numel(t);
states = zeros(n, 4);
u = zeros(n, 1);
failed = false;
failureTime = NaN;
states(1, :) = x0(:)';

for k = 1:n-1
    tk = t(k);
    xk = states(k, :)';
    uk = controller(tk, xk);
    if ~isfinite(uk)
        uk = 0;
    end
    uk = max(min(uk, params.u_max), -params.u_max);
    u(k) = uk;

    k1 = cartpole_dynamics(tk, xk, uk, params);
    k2 = cartpole_dynamics(tk + dt/2, xk + dt*k1/2, uk, params);
    k3 = cartpole_dynamics(tk + dt/2, xk + dt*k2/2, uk, params);
    k4 = cartpole_dynamics(tk + dt, xk + dt*k3, uk, params);
    nextState = xk + dt * (k1 + 2*k2 + 2*k3 + k4) / 6;
    states(k+1, :) = nextState';

    if any(~isfinite(nextState)) || abs(nextState(3)) > params.theta_fail
        failed = true;
        failureTime = t(k+1);
        states(k+2:end, :) = repmat(nextState', n-k-1, 1);
        u(k+1:end) = uk;
        break;
    end
end

u(end) = controller(t(end), states(end, :)');

result.label = char(label);
result.t = t;
result.x = states;
result.u = u;
result.failed = failed;
result.failureTime = failureTime;
result.dt = dt;
end
