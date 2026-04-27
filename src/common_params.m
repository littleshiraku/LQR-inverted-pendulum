function params = common_params()
%COMMON_PARAMS Shared constants and output paths for the cart-pole project.

thisFile = mfilename('fullpath');
srcDir = fileparts(thisFile);
rootDir = fileparts(srcDir);

params.rootDir = rootDir;
params.srcDir = srcDir;
params.figDir = fullfile(rootDir, 'figures');
params.animDir = fullfile(rootDir, 'animations');
params.dataDir = fullfile(rootDir, 'data');
params.logDir = fullfile(rootDir, 'data');

ensure_dir(params.figDir);
ensure_dir(params.animDir);
ensure_dir(params.dataDir);

params.m = 1.0;      % pendulum mass, kg
params.M = 5.0;      % cart mass, kg
params.L = 2.0;      % pendulum length to center of mass, m
params.g = 10.0;     % gravity, m/s^2

params.dt = 0.01;    % fixed zero-order-hold sample time, s
params.T = 30.0;     % default simulation horizon, s
params.x0 = [0; 0; 0.1; 0];

params.theta_fail = pi / 2;
params.theta_band = 0.01;
params.x_band = 0.02;
params.u_max = 200.0;

params.Q_base = diag([15, 5, 180, 50]);
params.R_base = 0.15;
end

function ensure_dir(pathName)
if ~exist(pathName, 'dir')
    mkdir(pathName);
end
end
