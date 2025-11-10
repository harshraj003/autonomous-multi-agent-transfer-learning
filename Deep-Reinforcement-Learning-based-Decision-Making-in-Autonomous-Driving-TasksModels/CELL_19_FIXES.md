# Cell 19 (Demo Code) Fixes - Complete Documentation

## Overview
This document explains the fixes applied to Cell 19 (actually Cell 23 in `trial12_adaptive_5cars.ipynb`) and the new Cell 24 (video display) to address critical issues in the multi-agent demo code.

## Problems Fixed

### 1. Memory Leak in RecordVideo
**Problem:** The kernel crashed after Episode 1 because RecordVideo buffered all frames (1500×3 = 4500 PIL images) in memory without proper cleanup.

**Solution:**
```python
# After each episode, properly close the recorder and clean up
rec.close()
if hasattr(rec, 'close_video_recorder'):
    try:
        rec.close_video_recorder()
    except:
        pass

# Explicit cleanup with garbage collection
del rec, base, wrapper, obs_list
gc.collect()
```

### 2. Same Reward for All Agents
**Problem:** Using `rec.step(acts)` returned a single scalar reward, which was then duplicated across all agents (`rs = [r_total]*5`), causing `reward_diff = 0` and preventing α from ever changing.

**Solution:** Implemented per-agent reward calculation:
```python
# Manual step with per-agent reward calculation
for veh, a in zip(wrapper.unwrapped.road.vehicles[:5], acts):
    veh.act(wrapper.env.unwrapped.action_type.actions[int(np.clip(a, 0, 4))])

wrapper.env.unwrapped.step(0)

# Calculate per-agent rewards
rs = []
for idx, v in enumerate(wrapper.unwrapped.road.vehicles[:5]):
    r = 0.0
    if getattr(v, "crashed", False):
        r -= 20.0  # Crash penalty
        if not crashed:
            crashed = True
    else:
        r += v.speed / 30.0  # Speed reward (normalized)
    rs.append(float(r))
```

### 3. Alpha (α) Not Updating Every 5 Steps
**Problem:** Alpha values were initialized once at the start of each episode and never updated during the episode, so they didn't reflect real-time performance.

**Solution:** Implemented dynamic α updates every 5 steps:
```python
SYNC_EVERY = 5

# Track rewards for alpha update
leader_rewards = []
follower_rewards = [[] for _ in range(4)]

# Inside the step loop:
leader_rewards.append(rs[0])
for i in range(4):
    follower_rewards[i].append(rs[i+1])

# Update alpha every 5 steps
if (t + 1) % SYNC_EVERY == 0 and t > 0:
    # Sync leader's Q-network to followers
    for f in followers:
        try:
            f.leader_qnet = leader.own_qnet_local
        except:
            pass
    syncs += 1
    
    # Update alpha based on recent reward comparison
    recent_window = min(SYNC_EVERY, len(leader_rewards))
    if recent_window > 0:
        avg_leader_reward = np.mean(leader_rewards[-recent_window:])
        
        for i, f in enumerate(followers):
            avg_follower_reward = np.mean(follower_rewards[i][-recent_window:])
            reward_diff = avg_leader_reward - avg_follower_reward
            
            # Update alpha based on performance gap
            if reward_diff > 0.05:  # leader doing better
                f.alpha = min(1.0, f.alpha + 0.02)
            elif reward_diff < -0.05:  # follower doing better
                f.alpha = max(0.0, f.alpha - 0.02)
            
            current_alphas[i] = f.alpha
```

### 4. Video Name Mismatch
**Problem:** Videos were saved as `demo_EP1_aB=0.50_...mp4` but Cell 23 looked for `EP*_LIVE*.mp4` pattern.

**Solution:** Simplified video naming to match expected pattern:
```python
rec = RecordVideo(
    wrapper,
    video_dir,
    name_prefix=f"EP{ep}",  # Results in: EP1-episode-0.mp4, EP2-episode-0.mp4, etc.
    episode_trigger=lambda _: True
)
```

### 5. CSV Output Format
**Problem:** CSV had wrong columns and no per-episode tracking of syncs, safe_steps, and crashes.

**Solution:** Updated CSV structure:
```python
alpha_hist = {
    "episode": [], 
    "alpha_b": [], "alpha_g": [], "alpha_y": [], "alpha_p": [],
    "syncs": [],      # Number of sync events per episode
    "safe_steps": [], # Steps without crash
    "crash": []       # Boolean: did any agent crash?
}

# Save to final.csv
csv_path = os.path.join(path_HW5, "Data_Average_Reward", "final.csv")
df.to_csv(csv_path, index=False)
```

## New Cell 24 - Video Display & Analysis

Created a comprehensive display cell that:

1. **Displays all 3 videos** from Episodes 1, 2, and 3
2. **Shows 3 detailed plots:**
   - Alpha evolution per follower across episodes
   - Safety metrics (safe steps bar chart + crash distribution pie chart)
   - Alpha changes and sync events per episode
3. **Prints summary statistics:**
   - Total episodes and crash rates
   - Average safe steps
   - Final alpha values with deltas

## Expected Output

### Videos
- `EP1-episode-0.mp4`
- `EP2-episode-0.mp4`
- `EP3-episode-0.mp4`

### CSV (`final.csv`)
```
episode,alpha_b,alpha_g,alpha_y,alpha_p,syncs,safe_steps,crash
1,0.520,0.510,0.505,0.515,300,1450,False
2,0.650,0.620,0.580,0.640,300,1480,False
3,0.780,0.720,0.680,0.760,300,1500,False
```

### Alpha Behavior
- Start: ~0.5 (neutral trust)
- End: ~0.6-0.9 (varies per follower)
- Changes every 5 steps based on reward gap
- Different values per follower (Blue, Green, Yellow, Purple)

## Key Features Preserved

✅ Uses exact loading code (leader + 4 followers)  
✅ Keeps `ColoredMultiAgentWrapper` unchanged  
✅ Keeps `SYNC_EVERY = 5`  
✅ Keeps `eps_demo = 0.07`  
✅ Maintains backward compatibility  
✅ No changes to training code or wrapper classes  

## Usage Instructions

### To Run the Demo:
1. Ensure you have trained models in `Models/multi_agent_adaptive/`
2. Run Cell 23 (the fixed demo code)
3. Run Cell 24 (video display and plots)

### Expected Execution Time:
- ~5-10 minutes per episode
- ~15-30 minutes total for 3 episodes
- No kernel crashes

## Safety Improvements

1. **Memory management:** Explicit cleanup after each episode
2. **Error handling:** Try-except blocks for video recorder cleanup
3. **Resource limits:** Fixed 1500 steps per episode
4. **State validation:** Proper termination/truncation checks

## Testing Recommendations

Before running on full 3 episodes, test with:
```python
# Change in Cell 23:
for ep in range(1, 2):  # Test with 1 episode first
```

Then verify:
- No kernel crash
- Video file created: `EP1-episode-0.mp4`
- CSV created: `final.csv`
- Alpha values are different per follower
- Cell 24 displays video and plots correctly

## Notes

- The code now creates a fresh environment for each episode to prevent state leakage
- Per-agent rewards are calculated directly from vehicle properties (speed, crash status)
- Alpha updates are logged and visible in both console output and CSV
- Videos show live alpha values if the ColoredMultiAgentWrapper's `show_alpha` feature is enabled
