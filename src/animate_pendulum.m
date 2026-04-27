function animate_pendulum(result, params, outPath, fps, maxDuration)
%ANIMATE_PENDULUM Save a 2-D cart-pole animation as GIF or AVI.

if nargin < 4 || isempty(fps)
    fps = 30;
end
if nargin < 5 || isempty(maxDuration)
    maxDuration = result.t(end);
end

frameStep = max(1, round(1 / (fps * params.dt)));
lastIdx = find(result.t <= maxDuration, 1, 'last');
if isempty(lastIdx)
    lastIdx = numel(result.t);
end
indices = 1:frameStep:lastIdx;

fig = figure('Color', 'w', 'Name', ['Animation: ', result.label], ...
    'Position', [120, 120, 900, 420]);
axis equal;
grid on;
xlabel('x (m)');
ylabel('y (m)');
ylim([-0.4, params.L + 0.5]);

xMin = min(result.x(indices, 1)) - params.L - 0.5;
xMax = max(result.x(indices, 1)) + params.L + 0.5;
xlim([xMin, xMax]);

cartW = 0.6;
cartH = 0.28;
wheelR = 0.06;

[~, ~, ext] = fileparts(outPath);
useGif = strcmpi(ext, '.gif');
if ~useGif
    writer = VideoWriter(outPath);
    writer.FrameRate = fps;
    open(writer);
end

for idx = indices
    cla;
    hold on;
    xCart = result.x(idx, 1);
    theta = result.x(idx, 3);
    pivot = [xCart, cartH/2];
    bob = [xCart + params.L*sin(theta), cartH/2 + params.L*cos(theta)];

    plot([xMin, xMax], [0, 0], 'k-', 'LineWidth', 1);
    rectangle('Position', [xCart-cartW/2, 0, cartW, cartH], ...
        'FaceColor', [0.30, 0.45, 0.70], 'EdgeColor', 'k');
    rectangle('Position', [xCart-cartW/3-wheelR, -wheelR, 2*wheelR, 2*wheelR], ...
        'Curvature', [1, 1], 'FaceColor', [0.1, 0.1, 0.1]);
    rectangle('Position', [xCart+cartW/3-wheelR, -wheelR, 2*wheelR, 2*wheelR], ...
        'Curvature', [1, 1], 'FaceColor', [0.1, 0.1, 0.1]);
    plot([pivot(1), bob(1)], [pivot(2), bob(2)], 'LineWidth', 4, 'Color', [0.85, 0.35, 0.15]);
    plot(pivot(1), pivot(2), 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 6);
    plot(bob(1), bob(2), 'o', 'MarkerFaceColor', [0.85, 0.15, 0.10], ...
        'MarkerEdgeColor', 'k', 'MarkerSize', 16);
    title(sprintf('%s, t = %.2f s', result.label, result.t(idx)), 'Interpreter', 'none');
    drawnow;

    frame = getframe(fig);
    if useGif
        [im, map] = rgb2ind(frame2im(frame), 256);
        if idx == indices(1)
            imwrite(im, map, outPath, 'gif', 'LoopCount', inf, 'DelayTime', 1/fps);
        else
            imwrite(im, map, outPath, 'gif', 'WriteMode', 'append', 'DelayTime', 1/fps);
        end
    else
        writeVideo(writer, frame);
    end
end

if ~useGif
    close(writer);
end
close(fig);
end
