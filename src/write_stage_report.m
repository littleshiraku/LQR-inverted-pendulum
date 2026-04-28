function write_stage_report(stageTitle, bodyText, params)
%WRITE_STAGE_REPORT Insert or replace one stage section in ppt.md.

pptPath = fullfile(params.rootDir, 'ppt.md');
if exist(pptPath, 'file')
    oldText = fileread(pptPath);
else
    oldText = sprintf(['# 阶段 1：开环验证\n\n', ...
        '# 阶段 2：LQR 闭环\n\n', ...
        '# 阶段 3：LLM 直接控制\n']);
end

stageTitle = char(stageTitle);
bodyText = char(bodyText);
sectionText = sprintf('# %s\n\n%s\n', stageTitle, bodyText);
header = ['# ', stageTitle];
startIdx = strfind(oldText, header);

if isempty(startIdx)
    if endsWith(oldText, newline)
        newText = [oldText, newline, sectionText];
    else
        newText = [oldText, newline, newline, sectionText];
    end
else
    startIdx = startIdx(1);
    tail = oldText(startIdx + length(header):end);
    nextRel = regexp(tail, '\n# ', 'once');
    if isempty(nextRel)
        sectionEnd = length(oldText) + 1;
    else
        sectionEnd = startIdx + length(header) + nextRel - 1;
    end
    newText = [oldText(1:startIdx-1), sectionText, oldText(sectionEnd:end)];
end

fid = fopen(pptPath, 'w', 'n', 'UTF-8');
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s', newText);
end
