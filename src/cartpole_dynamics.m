function dx = cartpole_dynamics(~, state, u, params)
%CARTPOLE_DYNAMICS Nonlinear upright cart-pole dynamics.
% State is [x; x_dot; theta; theta_dot], where theta=0 is upright.

xDot = state(2);
theta = state(3);
thetaDot = state(4);

m = params.m;
M = params.M;
L = params.L;
g = params.g;

den = M + m - m * cos(theta)^2;
xDDot = (u - m * g * sin(theta) * cos(theta) + ...
    m * L * sin(theta) * thetaDot^2) / den;
thetaDDot = (g * sin(theta) - cos(theta) * xDDot) / L;

dx = [xDot; xDDot; thetaDot; thetaDDot];
end
