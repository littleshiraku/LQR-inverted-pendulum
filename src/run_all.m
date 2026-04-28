%% run_all
% Execute the MATLAB-side workflow. Run run_stage3_llm_control.m separately
% after stage2 if Python LLM artifacts are required.

clear; clc;
run(fullfile('src', 'derive_model.m'));
run(fullfile('src', 'stage1_openloop.m'));
run(fullfile('src', 'stage2_lqr.m'));
run(fullfile('src', 'stage4_tuneQ.m'));

disp('MATLAB workflow complete. Run: run(''src/run_stage3_llm_control.m'') for Stage 3 LLM control.');
