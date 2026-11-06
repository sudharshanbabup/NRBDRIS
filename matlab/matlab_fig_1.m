function matlab_fig_1(ddir, odir)
% matlab_fig_1: MATLAB "Fig. 2d" style. One JSON per panel -> one vector PDF.
% Usage: matlab_fig_1            (reads ./data/*.json, writes ./figures/*.pdf)
%        matlab_fig_1(ddir, odir)
here = fileparts(mfilename('fullpath'));
if nargin < 1 || isempty(ddir); ddir = fullfile(here, 'data'); end
if nargin < 2 || isempty(odir); odir = fullfile(here, 'figures'); end
if ~exist(odir, 'dir'); mkdir(odir); end
files = dir(fullfile(ddir, '*.json'));
for k = 1:numel(files)
    S = jsondecode(fileread(fullfile(ddir, files(k).name)));
    draw_panel(S, odir);
end
fprintf('matlab_fig_1: wrote %d panel(s) to %s\n', numel(files), odir);
end

% ------------------------------------------------------------------------
function draw_panel(S, odir)
isOct = exist('OCTAVE_VERSION', 'builtin') > 0;
FN = pick_font(isOct);
lay = getf(S, 'layout', 'col');
% layout: [width height axisFont legendFont] in inches / pt
L = struct('col', [3.45 2.45 8 8], 'pair', [3.45 2.50 8 8], ...
           'trio', [2.36 2.10 7.5 6.5], 'quad', [1.77 1.90 7 6]);
p = L.(lay);
W = getf(S, 'width_in', p(1)); H = getf(S, 'height_in', p(2));
AFS = getf(S, 'font_pt', p(3)); LFS = getf(S, 'legend_pt', p(4));
sc = AFS / 8;                                   % shrink marks with font
LW = 1.5 * max(sc, 0.85); LWP = LW + 0.3; MS = 5 * max(sc, 0.85);
GREY = [0.40 0.40 0.40];
PAL = [0.000 0.447 0.698; 0.835 0.369 0.000; 0.000 0.620 0.451;
       0.482 0.196 0.580; 0.800 0.475 0.655; 0.902 0.624 0.000;
       0.337 0.706 0.914; 0.350 0.350 0.350];
MK = {'o', 's', 'd', '^', 'v', 'p', '>', 'h'};
LS = {'-', '-', '-', '-', '-', '-', '-', '-'};   % continuous lines only

fig = figure('Units', 'inches', 'Position', [1 1 W H], 'Color', 'w', ...
             'Visible', 'off', 'PaperUnits', 'inches', ...
             'PaperSize', [W H], 'PaperPosition', [0 0 W H]);
try; theme(fig, 'light'); catch; end            % ignore a dark desktop theme
ax = axes('Parent', fig); hold(ax, 'on');
set(ax, 'Color', 'w');
set(ax, 'FontName', FN, 'FontSize', AFS, 'LineWidth', 0.6, 'Box', 'on', ...
        'TickDir', 'in', 'Layer', 'top', 'XColor', 'k', 'YColor', 'k');
grid(ax, 'on');
try; set(ax, 'GridColor', [0.15 0.15 0.15], 'GridAlpha', 0.15, ...
             'MinorGridAlpha', 0.05); catch; end
try; set(ax, 'TickLength', [0.018 0.018]); catch; end
set_scales(ax, S);

series = S.series; if isstruct(series); series = num2cell(series); end
h = []; labs = {}; deferred = {}; nb = 0;
for i = 1:numel(series)
    s = series{i};
    kind = getf(s, 'kind', 'line');
    isProp = logical(getf(s, 'proposed', false));
    if ~any(strcmp(kind, {'ref', 'hline', 'vline', 'band'})); nb = nb + 1; end
    j = mod(max(nb, 1) - 1, size(PAL, 1)) + 1;
    c  = getf(s, 'color', PAL(j, :)); c = c(:)';
    mk = getf(s, 'marker', MK{j}); ls = getf(s, 'line', LS{j});
    if isempty(mk); mk = 'none'; end
    if ~strcmp(ls, 'none'); ls = '-'; end      % no dashed or dotted curves
    mfc = c; if getf(s, 'open', false); mfc = 'w'; end
    ms = MS; if any(strcmp(mk, {'d', '^', 'v', 'p', 'h', '>', '<'})); ms = MS + 0.8; end
    lw = LW; if isProp; lw = LWP; end
    lw = getf(s, 'lw', lw);
    hh = [];
    switch kind
        case 'line'
            hh = plot(ax, s.x, s.y, 'Color', c, 'LineStyle', ls, 'Marker', mk, ...
                'LineWidth', lw, 'MarkerSize', ms, 'MarkerFaceColor', mfc, ...
                'MarkerEdgeColor', c);
            if isfield(s, 'mstep')      % sparse markers on dense curves
                try; set(hh, 'MarkerIndices', 1:s.mstep:numel(s.x)); catch; end
            end
            if isfield(s, 'midx')       % explicit marker positions (1 based)
                try; set(hh, 'MarkerIndices', s.midx(:)'); catch; end
            end
        case 'errorbar'
            lo = getf(s, 'err_lo', getf(s, 'err', []));
            hi = getf(s, 'err_hi', getf(s, 'err', []));
            hh = errorbar(ax, s.x, s.y, lo, hi);
            set(hh, 'Color', c, 'LineStyle', ls, 'Marker', mk, 'LineWidth', lw, ...
                'MarkerSize', ms, 'MarkerFaceColor', mfc, 'MarkerEdgeColor', c);
            try; set(hh, 'CapSize', 3); catch; end
        case 'band'
            hh = fill(ax, [s.x(:); flipud(s.x(:))], [s.lo(:); flipud(s.hi(:))], c, ...
                'EdgeColor', 'none', 'FaceAlpha', getf(s, 'alpha', 0.15));
            s.label = '';
        case 'ref'          % dotted grey guide (O(N), bound, ...)
            cr = getf(s, 'color', GREY); cr = cr(:)';
            hh = plot(ax, s.x, s.y, 'Color', cr, 'LineStyle', '-', ...
                'LineWidth', getf(s, 'lw', 0.9), 'Marker', 'none');
        case 'bar'
            hh = bar(ax, s.x, s.y, getf(s, 'width', 0.7), 'FaceColor', c, ...
                'EdgeColor', 'k', 'LineWidth', 0.5);
        case {'hline', 'vline'}
            deferred{end + 1} = s; %#ok<AGROW>
            continue;
    end
    if ~isempty(getf(s, 'label', ''))
        h(end + 1) = hh(1); labs{end + 1} = s.label; %#ok<AGROW>
    end
end
if ~isfield(S, 'xlim'); xl = xlim(ax); xlim(ax, xl); end
if ~isfield(S, 'ylim')                % make room for horizontal solver lines
    yl = ylim(ax);
    for i = 1:numel(deferred)
        if strcmp(deferred{i}.kind, 'hline')
            yv = deferred{i}.y; yl = [min(yl(1), yv), max(yl(2), yv)];
            if strcmp(get(ax, 'YScale'), 'log'); yl = yl .* [0.9 1.1];
            else; yl = yl + [-0.03 0.03] * diff(yl); end
        end
    end
    ylim(ax, yl);
end
for i = 1:numel(deferred)            % solver / optimum lines across the axis
    s = deferred{i};
    c = getf(s, 'color', GREY); c = c(:)';
    if strcmp(s.kind, 'hline')
        hh = plot(ax, xlim(ax), [s.y s.y], 'Color', c, 'LineStyle', '-', ...
                  'LineWidth', getf(s, 'lw', 1.2), 'Marker', getf(s, 'marker', 'none'), ...
                  'MarkerFaceColor', c, 'MarkerEdgeColor', c, 'MarkerSize', MS);
        if ~strcmp(getf(s, 'marker', 'none'), 'none')   % a few markers along the level
            xl_ = xlim(ax);
            if strcmp(get(ax, 'XScale'), 'log'); xs_ = logspace(log10(xl_(1)), log10(xl_(2)), 7);
            else; xs_ = linspace(xl_(1), xl_(2), 7); end
            set(hh, 'XData', xs_(2:end-1), 'YData', s.y * ones(1, 5), 'LineStyle', '-');
            hh2 = plot(ax, xl_, [s.y s.y], 'Color', c, 'LineWidth', getf(s, 'lw', 1.2));
            uistack(hh2, 'bottom');
        end
    else
        hh = plot(ax, [s.x s.x], ylim(ax), 'Color', c, 'LineStyle', '-', ...
                  'LineWidth', getf(s, 'lw', 0.8), 'Tag', 'vline');
    end
    if ~isempty(getf(s, 'label', ''))
        h(end + 1) = hh; labs{end + 1} = s.label; %#ok<AGROW>
    end
end
xlabel(ax, getf(S, 'xlabel', ''), 'FontName', FN, 'FontSize', AFS, 'Color', 'k', 'Interpreter', 'tex');
ylabel(ax, getf(S, 'ylabel', ''), 'FontName', FN, 'FontSize', AFS, 'Color', 'k', 'Interpreter', 'tex');
if isfield(S, 'annotations')
    A = S.annotations; if isstruct(A); A = num2cell(A); end
    for i = 1:numel(A)
        a = A{i}; ca = getf(a, 'color', GREY);
        fa = 'normal'; if getf(a, 'italic', true); fa = 'italic'; end
        text(ax, a.x, a.y, a.text, 'FontName', FN, 'FontSize', AFS, ...
             'Color', ca(:)', 'FontAngle', fa, 'Interpreter', 'tex', ...
             'HorizontalAlignment', getf(a, 'halign', 'center'));
    end
end
fit_margins(fig, ax, S, W, H);
if ~isempty(h)
    lg = legend(ax, h, labs, 'Location', 'northeast');
    set(lg, 'FontSize', LFS, 'FontName', FN, 'Interpreter', 'tex');
    try; set(lg, 'Box', 'on', 'Color', 'w', 'EdgeColor', 'k', 'TextColor', 'k', 'LineWidth', 0.5); catch; end
    try; set(lg, 'ItemTokenSize', [15 * max(sc, 0.85), 8]); catch; end
    nc = getf(S, 'legend_cols', 1);
    if nc > 1; try; set(lg, 'NumColumns', nc); catch; end; end
    fit_legend(ax, lg, getf(S, 'legend', 'auto'), logical(getf(S, 'lock_ylim', false)));
end
stretch_vlines(ax); tick_format(ax, S);
fn = fullfile(odir, [S.name '.pdf']);
try; print(fig, fn, '-dpdf', '-vector'); catch; print(fig, fn, '-dpdf', '-painters'); end
if getf(S, 'png', false); print(fig, fullfile(odir, [S.name '.png']), '-dpng', '-r600'); end
close(fig); fprintf('  %s\n', S.name);
end

% ------------------------------------------------------------------------
function fit_margins(fig, ax, S, W, H)
% Fixed inch margins so every panel of a row has the same plot box; grow only
% if a tick label or axis label would be clipped.
m = getf(S, 'margins_in', [0.46 0.40 0.10 0.08]); m = m(:)';
set(ax, 'Units', 'inches');
for it = 1:3
    set(ax, 'Position', [m(1) m(2) W - m(1) - m(3) H - m(2) - m(4)]);
    drawnow;
    try; ti = get(ax, 'TightInset'); catch; break; end
    need = max(m, ti + 0.02);
    if all(need <= m + 1e-3); break; end
    m = need;
end
set(ax, 'Units', 'normalized');
set(fig, 'PaperPosition', [0 0 W H]);
end

function fit_legend(ax, lg, pref, lock)
% Put the legend in the emptiest corner; if every corner overlaps data, add
% headroom on the legend side until the legend sits fully inside, clear of data.
if strcmp(pref, 'auto')
    cand = {'northeast', 'northwest', 'southwest', 'southeast', 'north', 'south'};
else
    cand = {pref};
end
best = inf; bestc = cand{1};
for k = 1:numel(cand)
    set(lg, 'Location', cand{k}); drawnow;
    n = overlap(ax, lg);
    if n < best; best = n; bestc = cand{k}; end
    if n == 0; break; end
end
set(lg, 'Location', bestc); drawnow;
if lock; return; end
it = 0;
while overlap(ax, lg) > 0 && it < 10
    yl = ylim(ax); up = ~isempty(strfind(bestc, 'north'));
    if strcmp(get(ax, 'YScale'), 'log')
        r = 0.12 * log10(yl(2) / yl(1));
        if up; yl(2) = yl(2) * 10^r; else; yl(1) = yl(1) / 10^r; end
    else
        r = 0.12 * diff(yl);
        if up; yl(2) = yl(2) + r; else; yl(1) = yl(1) - r; end
    end
    ylim(ax, yl); set(lg, 'Location', bestc); drawnow; it = it + 1;
end
if it > 0; fprintf('    legend: added headroom (%d step(s))\n', it); end
end

function n = overlap(ax, lg)
% Count densified data samples (normalised axis units) under the legend box.
ap = get(ax, 'Position'); lp = get(lg, 'Position');
b = [(lp(1) - ap(1)) / ap(3), (lp(2) - ap(2)) / ap(4), lp(3) / ap(3), lp(4) / ap(4)];
pad = 0.035; n = 0;
hl = findobj(ax, '-property', 'YData');
ht = findobj(ax, 'Type', 'text');
for i = 1:numel(ht)                 % annotations count as occupied space
    try
        set(ht(i), 'Units', 'normalized'); e = get(ht(i), 'Extent'); set(ht(i), 'Units', 'data');
        [U, V] = meshgrid(linspace(e(1), e(1) + e(3), 4), linspace(e(2), e(2) + e(4), 3));
        n = n + sum(U(:) > b(1) - pad & U(:) < b(1) + b(3) + pad & ...
                    V(:) > b(2) - pad & V(:) < b(2) + b(4) + pad);
    catch
    end
end
for i = 1:numel(hl)
    if strcmp(get(hl(i), 'Type'), 'text'); continue; end
    x = double(get(hl(i), 'XData')); y = double(get(hl(i), 'YData'));
    x = x(:); y = y(:); if numel(x) ~= numel(y) || isempty(x); continue; end
    u = nrm(x, xlim(ax), get(ax, 'XScale'), get(ax, 'XDir'));
    v = nrm(y, ylim(ax), get(ax, 'YScale'), get(ax, 'YDir'));
    if numel(u) > 1
        t = linspace(0, 1, 25);
        U = []; V = [];
        for q = 1:numel(u) - 1
            U = [U, u(q) + t * (u(q + 1) - u(q))]; %#ok<AGROW>
            V = [V, v(q) + t * (v(q + 1) - v(q))]; %#ok<AGROW>
        end
        u = U; v = V;
    end
    in = u > b(1) - pad & u < b(1) + b(3) + pad & v > b(2) - pad & v < b(2) + b(4) + pad;
    n = n + sum(in(:));
end
end

function u = nrm(v, lim, scl, dr)
if strcmp(scl, 'log'); v = log10(max(v, realmin)); lim = log10(lim); end
u = (v - lim(1)) / (lim(2) - lim(1));
if strcmp(dr, 'reverse'); u = 1 - u; end
end

function tick_format(ax, S)
% fixed decimals on tick labels, e.g. "ytickformat": "%.1f" gives 0.0 0.2 ... 1.0
ax_ = {'x', 'y'};
for k = 1:2
    f = [ax_{k} 'tickformat'];
    if ~isfield(S, f); continue; end
    T = upper(ax_{k});
    t = get(ax, [T 'Tick']); set(ax, [T 'Tick'], t);
    lab = cell(1, numel(t));
    for q = 1:numel(t); lab{q} = sprintf(S.(f), t(q)); end
    set(ax, [T 'TickLabel'], lab);
end
end

function stretch_vlines(ax)
% vertical guides always span the final y range (after any legend headroom)
v = findobj(ax, 'Tag', 'vline');
for i = 1:numel(v); set(v(i), 'YData', ylim(ax)); end
end

function set_scales(ax, S)
if strcmp(getf(S, 'xscale', 'linear'), 'log'); set(ax, 'XScale', 'log'); end
if strcmp(getf(S, 'yscale', 'linear'), 'log'); set(ax, 'YScale', 'log'); end
if isfield(S, 'xlim'); xlim(ax, S.xlim(:)'); end
if isfield(S, 'ylim'); ylim(ax, S.ylim(:)'); end
if isfield(S, 'xticks'); set(ax, 'XTick', S.xticks(:)'); end
if isfield(S, 'xticklabels'); set(ax, 'XTickLabel', S.xticklabels); end
if isfield(S, 'yticks'); set(ax, 'YTick', S.yticks(:)'); end
if isfield(S, 'yticklabels'); set(ax, 'YTickLabel', S.yticklabels); end
if strcmp(getf(S, 'xdir', 'normal'), 'reverse'); set(ax, 'XDir', 'reverse'); end
set(ax, 'XMinorTick', 'off', 'YMinorTick', 'off');
try; set(ax, 'XMinorGrid', 'off', 'YMinorGrid', 'off'); catch; end
end

function FN = pick_font(isOct)
% Times, to match the IEEE body text
FN = 'Times New Roman';
if isOct; FN = 'FreeSerif'; return; end
try
    f = listfonts;
    if ~any(strcmpi(f, FN)); FN = 'Times'; end
catch
end
end

function v = getf(s, f, d)
if isfield(s, f) && ~isempty(s.(f)); v = s.(f); else; v = d; end
end
