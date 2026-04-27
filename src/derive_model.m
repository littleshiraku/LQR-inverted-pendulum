%% derive_model
% Symbolically derive and verify the nonlinear and linearized cart-pole model.

clear; clc;
params = common_params();
[A_expected, B_expected] = linear_matrices(params);

logPath = fullfile(params.logDir, 'derive_model_log.txt');
fid = fopen(logPath, 'w', 'n', 'UTF-8');
cleanup = onCleanup(@() fclose(fid));

fprintf(fid, 'Euler-Lagrange model derivation for upright cart-pole\n');
fprintf(fid, 'State: [x, x_dot, theta, theta_dot]^T\n\n');
fprintf(fid, 'Physical meaning and units:\n');
fprintf(fid, 'x: cart position, m\n');
fprintf(fid, 'x_dot: cart velocity, m/s\n');
fprintf(fid, 'theta: pendulum angle from upright vertical, rad\n');
fprintf(fid, 'theta_dot: angular velocity, rad/s\n\n');

fprintf(fid, 'Nonlinear equations:\n');
fprintf(fid, '(M+m)*x_ddot + m*L*cos(theta)*theta_ddot - m*L*sin(theta)*theta_dot^2 = u\n');
fprintf(fid, 'cos(theta)*x_ddot + L*theta_ddot - g*sin(theta) = 0\n\n');

symbolicVerified = false;
try
    syms m M L g x xdot theta thetadot u real
    den = M + m - m*cos(theta)^2;
    xddot = (u - m*g*sin(theta)*cos(theta) + m*L*sin(theta)*thetadot^2) / den;
    thetaddot = (g*sin(theta) - cos(theta)*xddot) / L;
    f = [xdot; xddot; thetadot; thetaddot];
    state = [x; xdot; theta; thetadot];
    A_sym = simplify(jacobian(f, state));
    B_sym = simplify(jacobian(f, u));
    A_eq = simplify(subs(A_sym, [x, xdot, theta, thetadot, u], [0, 0, 0, 0, 0]));
    B_eq = simplify(subs(B_sym, [x, xdot, theta, thetadot, u], [0, 0, 0, 0, 0]));

    A_formula = [0, 1, 0, 0;
                 0, 0, -m*g/M, 0;
                 0, 0, 0, 1;
                 0, 0, g*(M+m)/(M*L), 0];
    B_formula = [0; 1/M; 0; -1/(M*L)];

    symbolicVerified = isequal(simplify(A_eq - A_formula), sym(zeros(4))) && ...
        isequal(simplify(B_eq - B_formula), sym(zeros(4, 1)));
    fprintf(fid, 'Symbolic A matrix:\n%s\n', char(A_eq));
    fprintf(fid, 'Symbolic B matrix:\n%s\n', char(B_eq));
    fprintf(fid, 'Symbolic verification: %s\n\n', string(symbolicVerified));
catch err
    fprintf(fid, 'Symbolic Math Toolbox verification skipped or failed: %s\n\n', err.message);
end

A = A_expected;
B = B_expected;
openLoopPoles = eig(A);
save(fullfile(params.dataDir, 'model_matrices.mat'), ...
    'A', 'B', 'params', 'openLoopPoles', 'symbolicVerified');

fprintf(fid, 'Numeric A:\n');
fprintf(fid, '% .8f % .8f % .8f % .8f\n', A');
fprintf(fid, '\nNumeric B:\n');
fprintf(fid, '% .8f\n', B);
fprintf(fid, '\nOpen-loop poles:\n');
fprintf(fid, '% .8f%+ .8fi\n', [real(openLoopPoles), imag(openLoopPoles)]');

disp('Model matrices saved to data/model_matrices.mat');
disp(A);
disp(B);
