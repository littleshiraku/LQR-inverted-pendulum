%% run_all
% Execute the complete MATLAB-side workflow. Run stage3_llm_control.py separately
% with Python after stage2 if Python artifacts are required.

clear; clc;
run(fullfile('src', 'derive_model.m'));
run(fullfile('src', 'stage1_openloop.m'));
run(fullfile('src', 'stage2_lqr.m'));
run(fullfile('src', 'stage4_tuneQ.m'));

disp('MATLAB workflow complete. Run: python src/stage3_llm_control.py');
