%% train_stage3_lora_controller
% Train the Stage 3 LoRA controller through an external Python process.

clear; clc;

thisFile = mfilename('fullpath');
srcDir = fileparts(thisFile);
rootDir = fileparts(srcDir);
oldDir = pwd;
cleanup = onCleanup(@() cd(oldDir));
cd(rootDir);
setenv('PYTHONIOENCODING', 'utf-8');

pythonExe = getenv('STAGE3_PYTHON');
if isempty(pythonExe)
    pythonExe = 'python';
end

trainScript = fullfile(srcDir, 'train_stage3_lora_controller.py');
cmd = sprintf('"%s" "%s"', pythonExe, trainScript);

fprintf('Training Stage 3 LoRA controller with:\n%s\n\n', cmd);
[status, output] = system(cmd);
fprintf('%s\n', output);

if status ~= 0
    error('Stage 3 LoRA training failed with exit code %d.', status);
end

disp('Stage 3 LoRA training complete.');
