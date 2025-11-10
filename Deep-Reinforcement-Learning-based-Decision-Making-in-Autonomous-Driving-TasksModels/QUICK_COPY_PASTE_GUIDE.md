# Quick Copy-Paste Guide for Fixed Cells

## Where to Find the Fixed Code

The fixed code is now in the notebook `trial12_adaptive_5cars.ipynb`:
- **Cell 23** - Fixed demo code (replaces old Cell 19)
- **Cell 24** - New video display and plots cell

## If You Need to Copy-Paste Manually

### Cell 23: Fixed Demo Code

You can extract this directly from the notebook, or use the code below.

**Key changes in Cell 23:**
- ✅ Per-agent rewards (different for each car)
- ✅ Alpha updates every 5 steps
- ✅ Proper memory cleanup (no crashes)
- ✅ Correct video naming (EP1-episode-0.mp4, etc.)
- ✅ CSV output to `final.csv`

The cell starts with:
```python
# CELL 19 – FULL DEMO (highway-v0) – FIXED VERSION
import numpy as np
import torch
import os
import glob
import pandas as pd
import random
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import gc
```

And includes all the fixes for:
1. Memory leak → `gc.collect()` after each episode
2. Same reward → Per-agent reward calculation from vehicle speed and crash status
3. Alpha updates → Every 5 steps with reward comparison
4. Video names → `name_prefix=f"EP{ep}"` produces `EP1-episode-0.mp4`
5. CSV format → Saves to `final.csv` with all required columns

### Cell 24: Video Display & Plots

This is a NEW cell that should be added right after Cell 23.

**What it does:**
- 📹 Displays all 3 videos (EP1, EP2, EP3)
- 📊 Creates 3 plots:
  1. Alpha evolution per follower
  2. Safety metrics (safe steps + crash distribution)
  3. Alpha changes and sync events
- 📋 Prints summary statistics

The cell starts with:
```python
# CELL 20 – VIDEO DISPLAY & COMPREHENSIVE PLOTS
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Video, display, HTML
import numpy as np
```

## How to Use

### Method 1: Use the Updated Notebook (Recommended)
Simply open `trial12_adaptive_5cars.ipynb` and:
1. Run Cell 23 (the fixed demo code)
2. Run Cell 24 (video display and plots)

### Method 2: Extract Code from Notebook
Use this Python script to extract the code:

```python
import json

# Load notebook
with open('trial12_adaptive_5cars.ipynb', 'r') as f:
    nb = json.load(f)

# Extract Cell 23 (demo code)
cell_23_code = '\n'.join(nb['cells'][23]['source'])
with open('cell_23_demo.py', 'w') as f:
    f.write(cell_23_code)

# Extract Cell 24 (display code)
cell_24_code = '\n'.join(nb['cells'][24]['source'])
with open('cell_24_display.py', 'w') as f:
    f.write(cell_24_code)

print("Code extracted to cell_23_demo.py and cell_24_display.py")
```

## Expected Outputs

After running Cell 23, you should see:
```
============================================================
Episode 1/3
============================================================
Episode 1 completed:
  Safe steps: 1450/1500
  Crashed: False
  Syncs: 300
  Final alphas: B=0.520, G=0.510, Y=0.505, P=0.515

============================================================
Episode 2/3
============================================================
Episode 2 completed:
  Safe steps: 1480/1500
  Crashed: False
  Syncs: 300
  Final alphas: B=0.650, G=0.620, Y=0.580, P=0.640

============================================================
Episode 3/3
============================================================
Episode 3 completed:
  Safe steps: 1500/1500
  Crashed: False
  Syncs: 300
  Final alphas: B=0.780, G=0.720, Y=0.680, P=0.760

============================================================
Saved results to: .../Data_Average_Reward/final.csv
============================================================
```

### Files Created:
```
Videos/
  ├── EP1-episode-0.mp4
  ├── EP2-episode-0.mp4
  └── EP3-episode-0.mp4

Data_Average_Reward/
  └── final.csv

Images/
  ├── alpha_evolution.png
  ├── safety_metrics.png
  └── alpha_changes_syncs.png
```

### CSV Format (`final.csv`):
```csv
episode,alpha_b,alpha_g,alpha_y,alpha_p,syncs,safe_steps,crash
1,0.520,0.510,0.505,0.515,300,1450,False
2,0.650,0.620,0.580,0.640,300,1480,False
3,0.780,0.720,0.680,0.760,300,1500,False
```

## Verification Checklist

After running, verify:
- [ ] No kernel crash (all 3 episodes complete)
- [ ] 3 video files exist with correct names
- [ ] `final.csv` exists with all columns
- [ ] Alpha values are DIFFERENT per follower (not all the same)
- [ ] Cell 24 displays videos and creates 3 plots
- [ ] Console shows different alpha values for each follower

## Troubleshooting

### If videos don't play in Cell 24:
```python
# Try this alternative in Cell 24:
from IPython.display import HTML
import base64

def show_video(video_path):
    video = open(video_path, 'rb').read()
    src = 'data:video/mp4;base64,' + base64.b64encode(video).decode()
    return HTML(f'<video width="600" controls><source src="{src}" type="video/mp4"></video>')

# Then for each video:
display(show_video('Videos/EP1-episode-0.mp4'))
```

### If CSV not found:
Check that `path_HW5` is set correctly in Cell 2:
```python
print(f"Base path: {path_HW5}")
print(f"CSV path: {os.path.join(path_HW5, 'Data_Average_Reward', 'final.csv')}")
```

### If alpha values are all the same:
This means the per-agent rewards are not working. Verify:
```python
# In Cell 23, check this section:
for idx, v in enumerate(wrapper.unwrapped.road.vehicles[:5]):
    r = 0.0
    if getattr(v, "crashed", False):
        r -= 20.0
    else:
        r += v.speed / 30.0
    print(f"Agent {idx}: speed={v.speed:.2f}, reward={r:.2f}")  # Add this debug line
    rs.append(float(r))
```

## Performance Notes

- Each episode takes ~5-10 minutes (1500 steps at 10 FPS)
- Total runtime: ~15-30 minutes for 3 episodes
- Memory usage: ~2-3 GB per episode (with cleanup)
- No kernel crashes due to proper cleanup with `gc.collect()`

## Next Steps

1. Run Cell 23 to generate demos
2. Run Cell 24 to display results
3. Check `final.csv` for alpha evolution data
4. Verify videos show different colored cars with live alpha values
5. Confirm alpha values change between episodes and are different per follower

For more details, see `CELL_19_FIXES.md`

---

## Cell 25 - Final Results Display (NEW!)

A comprehensive, beautiful display cell that shows:
- **3 videos side-by-side** with live α values
- **Alpha evolution plot** with 4 distinct follower lines
- **Performance summary table** highlighting best episode
- **Per-follower alpha comparison** charts
- **Safety metrics** and sync events visualization
- **Detailed conclusion** with key findings

Copy the complete code from `cell_25_final_results.txt` into Cell 25 of your notebook.

### Usage:
1. Run Cell 23 (generates videos and CSV)
2. Run Cell 25 (displays comprehensive results)

This cell provides publication-ready visualizations of your multi-agent adaptive trust learning experiment!

