import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import importlib

# Ensure the workspace root is on sys.path
import sys
from pathlib import Path as P
workspace_root = P(__file__).resolve().parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

# Import the plotting module
try:
    import full_link_nrz_stateye as fl
except Exception as e:
    print("Error importing full_link_part3:", e)
    raise

# Override plt.show to save all figures to plots/ directory
def save_and_close(*args, **kwargs):
    outdir = workspace_root / 'plots'
    outdir.mkdir(exist_ok=True)
    saved = []
    import time
    ts = time.strftime('%Y%m%d_%H%M%S')
    for i, num in enumerate(plt.get_fignums(), start=1):
        fig = plt.figure(num)
        # Try to build a meaningful filename from figure title
        title = None
        try:
            if fig.axes and len(fig.axes) > 0:
                title = fig.axes[0].get_title()
        except Exception:
            title = None

        def _safe(s):
            import re
            s = s.strip()
            s = re.sub(r'\s+', '_', s)
            s = re.sub(r'[^A-Za-z0-9_\-]', '', s)
            return s or None

        safe_title = _safe(title) if title else None
        base = safe_title if safe_title else f'plot_{i}'
        # avoid overwriting by appending timestamp
        filename = f"{base}_{ts}.png"
        path = outdir / filename
        fig.savefig(path, dpi=200)
        saved.append(str(path))
    print('Saved figures:')
    for s in saved:
        print('  ', s)
    plt.close('all')

plt.show = save_and_close

if __name__ == '__main__':
    fl.main()
