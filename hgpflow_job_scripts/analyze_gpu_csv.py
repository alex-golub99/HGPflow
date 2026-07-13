"""
Separate GPU-idle (loading/idle) phases from the active training window in a
gpu_profile.sh CSV, and report utilization stats for the training window only.

Usage:
    python analyze_gpu_csv.py gpu_util.csv [mem_active_threshold_MiB]
"""
import sys
import numpy as np

csv_path = sys.argv[1] if len(sys.argv) > 1 else 'gpu_util.csv'
mem_thresh = float(sys.argv[2]) if len(sys.argv) > 2 else 500.0
DT = 0.5  # sampler interval (seconds); matches gpu_profile.sh default

rows = []
with open(csv_path) as f:
    next(f)  # header
    for line in f:
        p = line.strip().split(',')
        if len(p) < 4:
            continue
        try:
            rows.append((float(p[1]), float(p[3])))  # gpu_util, mem_used
        except ValueError:
            continue

rows = np.array(rows)
util, mem = rows[:, 0], rows[:, 1]
n = len(rows)
active = mem > mem_thresh
idx = np.where(active)[0]

print(f'total samples      : {n}  (~{n*DT/60:.1f} min)')
print(f'active (mem>{mem_thresh:.0f}MiB): {active.sum()}  (~{active.sum()*DT/60:.1f} min)')
print(f'idle / loading     : {n-active.sum()}  (~{(n-active.sum())*DT/60:.1f} min)')

if len(idx):
    # training window = span from first to last active sample
    lo, hi = idx[0], idx[-1]
    win = util[lo:hi+1]
    print(f'\ntraining window    : ~{lo*DT/60:.1f} -> {hi*DT/60:.1f} min '
          f'({(hi-lo+1)*DT:.0f}s wall)')
    print('--- GPU util in training window ---')
    print(f'  mean   : {win.mean():.1f} %')
    print(f'  median : {np.median(win):.0f} %')
    print(f'  p95    : {np.percentile(win, 95):.0f} %')
    print(f'  max    : {win.max():.0f} %')
    print(f'  frac at 0% util (starved) : {(win == 0).mean()*100:.0f}%')
    print(f'\n  peak mem used : {mem.max():.0f} MiB')
    verdict = ('GPU-bound (good)' if win.mean() > 85 else
               'partially starved' if win.mean() > 50 else
               'CPU/data-starved (sawtooth)')
    print(f'  verdict: {verdict}')
else:
    print('\nno active (mem>threshold) samples found - lower the threshold?')
