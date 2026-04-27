function [A, B] = linear_matrices(params)
%LINEAR_MATRICES Continuous-time linearization at the upright equilibrium.

m = params.m;
M = params.M;
L = params.L;
g = params.g;

A = [0, 1, 0, 0;
     0, 0, -m*g/M, 0;
     0, 0, 0, 1;
     0, 0, g*(M+m)/(M*L), 0];

B = [0;
     1/M;
     0;
    -1/(M*L)];
end
