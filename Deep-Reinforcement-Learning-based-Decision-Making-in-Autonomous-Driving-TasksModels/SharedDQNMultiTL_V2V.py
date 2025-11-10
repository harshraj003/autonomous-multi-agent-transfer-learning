"""
Enhanced Multi-Agent DQN Trainer with V2V Communication for CACC in Connected Cars

This module implements Vehicle-to-Vehicle (V2V) communication for multi-agent DRL
using a similarity coefficient mechanism. Following agents learn from preceding agents
by computing state similarity and blending actions accordingly.

Author: Enhanced CACC Framework
Date: 2024
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any
from tqdm import trange
from scipy.spatial.distance import cosine
from collections import deque, namedtuple


class SharedDQNMultiTL:
    """
    Multi-agent DQN trainer with Vehicle-to-Vehicle (V2V) communication mechanism.
    
    Architecture:
    - Shared QNetwork across all agents (weight sharing for cooperative learning)
    - Shared ReplayBuffer for experience aggregation
    - V2V similarity mechanism: agent i learns from agent i-1
    - Similarity-based action weighting and reward shaping
    
    Key Features:
    1. Similarity Computation:
       - Cosine or Euclidean distance between agent and preceding agent states
       - Returns coefficient in [0, 1]
    
    2. Action Selection with V2V:
       - High similarity (≥0.8): 85% follow preceding agent
       - Medium similarity (0.4-0.8): 50% blend based on similarity
       - Low similarity (<0.4): Independent greedy action
    
    3. Reward Shaping:
       - Safe gap bonus: +0.5 if gap ≥ 5.0m
       - Close gap penalty: -0.5 if gap < 2.0m
       - Encourages safe platoon formation
    
    4. Efficiency:
       - Single shared network for 4 agents
       - Reduced parameter count vs. independent agents
       - Faster convergence through weight sharing
    """
    
    def __init__(self, env, network_type: str, device: torch.device, seed: int = 11,
                 use_v2v: bool = True, similarity_metric: str = 'cosine',
                 v2v_weights: dict = None):
        """
        Initialize the multi-agent CACC trainer.
        
        Args:
            env: Gymnasium environment (highway-v0 with 4 controlled vehicles)
            network_type: 'linear' or 'cnn' for network architecture
            device: torch.device (cuda or cpu)
            seed: Random seed for reproducibility
            use_v2v: Enable V2V learning mechanism (default: True)
            similarity_metric: 'cosine' or 'euclidean' for state comparison
            v2v_weights: Optional dict overriding default V2V parameters
        """
        self.env = env
        self.device = device
        self.network_type = network_type
        self.use_v2v = use_v2v
        self.similarity_metric = similarity_metric
        
        # V2V Configuration: Tunable parameters for reward shaping and action blending
        self.v2v_config = v2v_weights or {
            'high_sim_weight': 0.85,           # Probability to follow preceding agent when similarity >= 0.8
            'medium_sim_weight': 0.50,         # Probability to follow preceding agent when 0.4 <= similarity < 0.8
            'gap_reward': 0.5,                 # Bonus reward for maintaining safe gap
            'collision_penalty': -0.5,         # Penalty for driving too close
            'safe_gap_threshold': 5.0,         # Distance threshold for "safe gap" (meters)
            'collision_threshold': 2.0,        # Distance threshold for "too close" (meters)
        }
        
        # Probe environment to determine sizes
        obs_any, _ = self.env.reset()
        obs_dict = self._unify_obs_dict(obs_any)
        self.num_agents = self.env.unwrapped.config.get("controlled_vehicles", len(obs_dict))
        sample_obs = next(iter(obs_dict.values()))
        self.state_size = int(np.prod(sample_obs.shape))
        
        # Infer action space size
        sp = getattr(self.env, "action_space", None)
        if hasattr(sp, "spaces"):
            self.action_size = next(iter(sp.spaces.values())).n
        elif isinstance(sp, (list, tuple)):
            self.action_size = sp[0].n
        else:
            self.action_size = sp.n
        
        # Import Agent class (must be available in environment)
        from __main__ import Agent as AgentClass
        self.agent = AgentClass(self.state_size, self.action_size, network_type, seed)
        
        # V2V State Memory: Track preceding agent info for similarity computation
        self.prev_obs = [None] * self.num_agents           # Previous observation per agent
        self.prev_actions = [None] * self.num_agents       # Previous action per agent
        self.agent_gaps = [0.0] * self.num_agents          # Track inter-agent gaps

    def _compute_similarity(self, state1: np.ndarray, state2: np.ndarray) -> float:
        """
        Compute state similarity using cosine or Euclidean distance.
        
        Cosine Similarity:
        - Measure angle between state vectors
        - Robust to magnitude differences
        - Good for high-dimensional states
        
        Euclidean Similarity:
        - Normalized distance metric
        - Sensitive to all dimension changes
        - Alternative for comparison
        
        Args:
            state1, state2: State vectors (flattened numpy arrays)
            
        Returns:
            Similarity coefficient in [0, 1] (1 = identical, 0 = completely different)
        """
        if state1 is None or state2 is None:
            return 0.0
        
        s1 = state1.ravel()
        s2 = state2.ravel()
        
        # Shape mismatch: return 0 similarity
        if s1.shape != s2.shape:
            return 0.0
        
        if self.similarity_metric == 'cosine':
            try:
                # Cosine distance: dist = 1 - (dot(A,B) / (||A||*||B||))
                # Convert to similarity: sim = 1 - dist
                dist = cosine(s1, s2)
                return float(np.clip(1.0 - dist, 0.0, 1.0))
            except Exception:
                return 0.0
        else:  # euclidean
            # Normalized Euclidean: similarity = 1 - (dist / max_possible_dist)
            norm_s1 = np.linalg.norm(s1)
            norm_s2 = np.linalg.norm(s2)
            if norm_s1 < 1e-6 or norm_s2 < 1e-6:
                return 1.0 if np.allclose(s1, s2) else 0.0
            eucl_dist = np.linalg.norm(s1 - s2)
            max_dist = norm_s1 + norm_s2  # Upper bound on distance
            return float(np.clip(1.0 - (eucl_dist / max_dist), 0.0, 1.0))

    def _extract_gap_from_obs(self, obs: np.ndarray, agent_idx: int) -> float:
        """
        Extract longitudinal gap (distance to preceding vehicle) from kinematics observation.
        
        Observation Format:
        Shape: (vehicles_count, 5) with features [presence, x, y, vx, vy]
        - presence: 1 if vehicle exists, 0 otherwise
        - x, y: Absolute position (or relative if "absolute": False in config)
        - vx, vy: Velocity components
        
        Gap Calculation:
        - Current agent at obs[agent_idx, 1] (x position)
        - Preceding agent at obs[agent_idx-1, 1] (x position)
        - Gap = x_preceding - x_current (positive if preceding is ahead)
        
        Args:
            obs: Observation array
            agent_idx: Current agent index
            
        Returns:
            Gap distance in meters (0 if N/A)
        """
        try:
            if obs is None or obs.size == 0:
                return 0.0
            
            # Reshape observation to 2D: (vehicles_count, features)
            obs_2d = obs.reshape(-1, 5) if obs.ndim == 1 else obs
            
            if agent_idx > 0 and agent_idx < len(obs_2d):
                # Current agent's x position
                my_x = obs_2d[agent_idx, 1] if agent_idx < len(obs_2d) else 0.0
                # Preceding agent's x position
                prev_x = obs_2d[agent_idx - 1, 1] if (agent_idx - 1) < len(obs_2d) else 0.0
                gap = prev_x - my_x
                return float(max(0.0, gap))  # Return gap (0 if too close)
            return 0.0
        except Exception:
            return 0.0

    def _compute_v2v_reward(self, obs: np.ndarray, agent_idx: int, base_reward: float) -> float:
        """
        Compute V2V-aware reward with platoon-specific bonuses/penalties.
        
        Reward Structure:
        1. Base reward from environment (collision penalty, speed bonus, etc.)
        2. + Safe gap bonus: agent maintains distance >= 5.0m from preceding vehicle
        3. - Collision penalty: agent too close (<2.0m) to preceding vehicle
        
        This shapes the reward to encourage safe CACC behavior:
        - Maintains appropriate gap for emergency braking
        - Penalizes dangerous following distances
        - Only applies to agents i > 0 (followers in platoon)
        
        Args:
            obs: Current observation
            agent_idx: Agent index
            base_reward: Reward from environment
            
        Returns:
            Modified reward with V2V bonuses/penalties
        """
        # Only apply V2V reward to following agents (not lead agent)
        if agent_idx == 0 or not self.use_v2v:
            return base_reward
        
        gap = self._extract_gap_from_obs(obs, agent_idx)
        cfg = self.v2v_config
        
        v2v_bonus = 0.0
        if gap >= cfg['safe_gap_threshold']:
            # Safe gap maintained: encourage this behavior
            v2v_bonus += cfg['gap_reward']
        elif gap < cfg['collision_threshold']:
            # Too close: penalize
            v2v_bonus -= cfg['collision_penalty']
        
        self.agent_gaps[agent_idx] = gap
        return base_reward + v2v_bonus

    def _get_v2v_weighted_action(self, agent_idx: int, greedy_action: int,
                                 preceding_action: int, similarity: float) -> int:
        """
        Blend preceding agent's action with greedy action based on similarity coefficient.
        
        Mechanism:
        1. High Similarity (≥0.8): Follow preceding agent closely
           - 85% chance to execute preceding agent's action
           - 15% chance to try own greedy action (exploration)
        
        2. Medium Similarity (0.4-0.8): Adaptively blend
           - Probability = medium_sim_weight (50% by default)
           - Balanced between cooperation and individual learning
        
        3. Low Similarity (<0.4): Independent learning
           - Use own greedy action
           - Preceding agent's state is too different to learn from
        
        This mimics real CACC: vehicles follow closely when traffic is similar,
        but maintain independence when conditions differ significantly.
        
        Args:
            agent_idx: Current agent index
            greedy_action: Agent's own greedy action from Q-network
            preceding_action: Preceding agent's previous action
            similarity: Similarity coefficient between states [0, 1]
            
        Returns:
            Selected action (int)
        """
        if agent_idx == 0 or not self.use_v2v or preceding_action is None:
            # Lead agent or V2V disabled: use greedy action
            return greedy_action
        
        if similarity >= self.v2v_config['high_sim_weight']:
            # High similarity (≥0.85): Follow preceding agent with high probability
            return preceding_action if np.random.rand() < 0.85 else greedy_action
        
        elif similarity >= 0.4:
            # Medium similarity (0.4-0.85): Blend based on configured weight
            weight = self.v2v_config['medium_sim_weight']
            return preceding_action if np.random.rand() < weight else greedy_action
        
        else:
            # Low similarity (<0.4): Independent learning
            return greedy_action

    @staticmethod
    def _unify_obs_dict(obs_any) -> Dict[int, np.ndarray]:
        """
        Convert various observation formats to unified dict format.
        
        Handles:
        - Dict format (keyed by agent index)
        - List/Tuple format (indexed by agent)
        - 2D array format (rows are agents)
        """
        if isinstance(obs_any, dict):
            return {int(k): np.asarray(v) for k, v in obs_any.items()}
        if isinstance(obs_any, (list, tuple)):
            return {i: np.asarray(obs_any[i]) for i in range(len(obs_any))}
        if isinstance(obs_any, np.ndarray) and obs_any.ndim >= 2:
            return {i: obs_any[i] for i in range(obs_any.shape[0])}
        raise RuntimeError(f"Unexpected observation format: {type(obs_any)}")

    @staticmethod
    def _unify_to_list(x_any, n: int) -> List[Any]:
        """Convert various response formats to list format."""
        if isinstance(x_any, dict):
            return [x_any[i] for i in range(n)]
        if isinstance(x_any, (list, tuple)):
            return list(x_any)
        if isinstance(x_any, np.ndarray):
            return x_any.tolist()
        return [x_any] * n

    def _act_batch(self, states: List[np.ndarray], eps: float) -> List[int]:
        """
        Compute batch actions for all agents with V2V learning.
        
        Process:
        1. Stack all agent states → single tensor
        2. Forward pass through shared Q-network → Q-values for all agents
        3. Epsilon-greedy selection (exploration vs. exploitation)
        4. V2V weighting: blend preceding agent's action for followers
        
        Args:
            states: List of observation arrays, one per agent
            eps: Exploration rate for epsilon-greedy
            
        Returns:
            List of actions, one per agent
        """
        # Stack states: (num_agents, state_size)
        st = torch.from_numpy(np.stack([s.ravel() for s in states], axis=0)).float().to(self.device)
        
        # Inference: Q-network forward pass
        self.agent.qnetwork_local.eval()
        with torch.no_grad():
            q = self.agent.qnetwork_local(st)  # Shape: (num_agents, action_size)
        self.agent.qnetwork_local.train()
        
        # Greedy actions (argmax per agent)
        greedy = q.argmax(dim=1).cpu().numpy()
        
        # Epsilon-greedy: explore vs. exploit
        mask = np.random.rand(self.num_agents) < eps
        rand = np.random.randint(0, self.action_size, size=self.num_agents)
        
        actions = []
        for i in range(self.num_agents):
            # Step 1: Epsilon-greedy selection
            base_action = int(rand[i]) if mask[i] else int(greedy[i])
            
            # Step 2: V2V weighting for followers (i > 0)
            if i > 0 and self.use_v2v and self.prev_obs[i-1] is not None:
                similarity = self._compute_similarity(states[i], self.prev_obs[i-1])
                preceding_action = self.prev_actions[i-1]
                base_action = self._get_v2v_weighted_action(i, base_action, preceding_action, similarity)
            
            actions.append(base_action)
        
        return actions

    def load_transfer_weights(self, checkpoint_path: str) -> None:
        """
        Load pre-trained weights from single-agent model (transfer learning).
        
        Enables warm-start training:
        - Start multi-agent training with weights from single-agent merge-v0
        - Reduces training time significantly
        - Initializes Q-network with reasonable value estimates
        
        Args:
            checkpoint_path: Path to saved model checkpoint
        """
        if not os.path.exists(checkpoint_path):
            print(f"Warning: checkpoint not found at {checkpoint_path}")
            return
        
        sd = torch.load(checkpoint_path, map_location=self.device)
        missing, unexpected = self.agent.qnetwork_local.load_state_dict(sd, strict=False)
        print(f"Transfer load complete: {len(missing)} missing keys, {len(unexpected)} unexpected")

    def train(self, runs: List[int], episodes_per_run: int, max_steps: int,
              out_models: str, out_data: str, save_every: int = 100,
              eps_start=1.0, eps_end=0.01, eps_decay=0.995) -> pd.DataFrame:
        """
        Train multi-agent CACC system with V2V communication.
        
        Training Flow Per Episode:
        1. Reset environment and V2V state memory
        2. For each step:
           a. Compute V2V-weighted actions for all agents
           b. Execute step in environment
           c. Compute V2V-shaped rewards
           d. Store experiences in shared replay buffer (with V2V rewards)
           e. Learn from batch samples (DQN update)
        3. Save checkpoint every save_every episodes
        4. Decay epsilon for exploration schedule
        
        Key Features:
        - Shared replay buffer: all agents contribute to single pool
        - Shared Q-network: weight sharing for efficiency
        - V2V reward shaping: encourages safe platoon formation
        - Transfer learning: warm-start from single-agent checkpoints
        
        Args:
            runs: List of run numbers to train
            episodes_per_run: Number of episodes per run
            max_steps: Maximum steps per episode
            out_models: Directory to save model checkpoints
            out_data: Directory to save reward statistics
            save_every: Save checkpoints every N episodes
            eps_start: Initial exploration rate
            eps_end: Final exploration rate
            eps_decay: Exponential decay factor per episode
            
        Returns:
            DataFrame with per-agent reward statistics
        """
        os.makedirs(out_models, exist_ok=True)
        os.makedirs(out_data, exist_ok=True)
        df = pd.DataFrame()
        
        for r in runs:
            eps = eps_start
            rewards_agents: List[List[float]] = [[] for _ in range(self.num_agents)]
            
            for ep in trange(1, episodes_per_run+1, desc=f"MA-V2V Run {r}"):
                # Reset episode
                obs_any, _ = self.env.reset()
                obs_dict = self._unify_obs_dict(obs_any)
                self.prev_obs = [None] * self.num_agents
                self.prev_actions = [None] * self.num_agents
                self.agent_gaps = [0.0] * self.num_agents
                
                done = {i: False for i in range(self.num_agents)}
                totals = [0.0] * self.num_agents
                
                # Episode steps
                for step in range(max_steps):
                    obs_list = [obs_dict[i] for i in range(self.num_agents)]
                    actions = self._act_batch(obs_list, eps)
                    
                    # Environment step
                    try:
                        nxt, rews, terms, truncs, infos = self.env.step(
                            {i: actions[i] for i in range(self.num_agents)}
                        )
                    except Exception:
                        nxt, rews, terms, truncs, infos = self.env.step(actions)
                    
                    nxt_dict = self._unify_obs_dict(nxt)
                    rews_l = self._unify_to_list(rews, self.num_agents)
                    terms_l = self._unify_to_list(terms, self.num_agents)
                    truncs_l = self._unify_to_list(truncs, self.num_agents)
                    
                    # Process each agent
                    for i in range(self.num_agents):
                        d = bool(terms_l[i] or truncs_l[i])
                        base_rew = float(rews_l[i])
                        
                        # V2V Reward Shaping
                        v2v_rew = self._compute_v2v_reward(obs_list[i], i, base_rew)
                        
                        # Store in shared replay buffer
                        self.agent.step(obs_list[i].ravel(), actions[i], v2v_rew,
                                       nxt_dict[i].ravel(), d)
                        totals[i] += base_rew  # Track base reward for reporting
                        done[i] = d
                        
                        # Update V2V state for next step
                        self.prev_obs[i] = obs_list[i].copy()
                        self.prev_actions[i] = actions[i]
                    
                    obs_dict = nxt_dict
                    if all(done.values()):
                        break
                
                # Record episode rewards
                for i in range(self.num_agents):
                    rewards_agents[i].append(totals[i])
                
                # Decay exploration rate
                eps = max(eps_end, eps * eps_decay)
                
                # Save checkpoint
                if ep % save_every == 0:
                    rd = Path(out_models) / f"Models_Run_{r}"
                    rd.mkdir(parents=True, exist_ok=True)
                    torch.save(self.agent.qnetwork_local.state_dict(),
                              rd / f"ma_v2v_local_{ep}.pth")
                    torch.save(self.agent.qnetwork_target.state_dict(),
                              rd / f"ma_v2v_target_{ep}.pth")
            
            # Save rewards to CSV
            for i in range(self.num_agents):
                df[f"Run_{r}_Agent_{i}"] = rewards_agents[i]
            df.to_csv(os.path.join(out_data, f"Episodes_reward_run_{r}.csv"), index=False)
        
        return df

    def evaluate_and_record(self, video_dir: str, episodes: int = 3) -> Tuple[List[float], List[float]]:
        """
        Evaluate trained multi-agent system with video recording.
        
        Evaluation:
        - Uses greedy policy (eps=0.0, no exploration)
        - Records video of episodes
        - Tracks reward and crash rate per agent
        
        Args:
            video_dir: Directory to save evaluation videos
            episodes: Number of episodes to evaluate
            
        Returns:
            (avg_rewards_per_agent, crash_rates_per_agent)
        """
        os.makedirs(video_dir, exist_ok=True)
        from gymnasium.wrappers import RecordVideo
        env_wrapped = RecordVideo(self.env, video_folder=video_dir,
                                 name_prefix="ma_v2v_eval",
                                 episode_trigger=lambda e: True)
        totals = [0.0] * self.num_agents
        crashes = [0] * self.num_agents
        
        for _ in range(episodes):
            obs_any, _ = env_wrapped.reset()
            obs_dict = self._unify_obs_dict(obs_any)
            
            # Reset V2V memory
            self.prev_obs = [None] * self.num_agents
            self.prev_actions = [None] * self.num_agents
            self.agent_gaps = [0.0] * self.num_agents
            
            done = {i: False for i in range(self.num_agents)}
            
            while not all(done.values()):
                obs_list = [obs_dict[i] for i in range(self.num_agents)]
                actions = self._act_batch(obs_list, eps=0.0)
                
                # Step
                try:
                    nxt, rews, terms, truncs, infos = env_wrapped.step(
                        {i: actions[i] for i in range(self.num_agents)}
                    )
                except Exception:
                    nxt, rews, terms, truncs, infos = env_wrapped.step(actions)
                
                nxt_dict = self._unify_obs_dict(nxt)
                rews_l = self._unify_to_list(rews, self.num_agents)
                terms_l = self._unify_to_list(terms, self.num_agents)
                truncs_l = self._unify_to_list(truncs, self.num_agents)
                infos_l = self._unify_to_list(infos, self.num_agents)
                
                for i in range(self.num_agents):
                    totals[i] += float(rews_l[i])
                    info_i = infos_l[i] if isinstance(infos_l[i], dict) else {}
                    if isinstance(info_i, dict) and info_i.get("crashed", False):
                        crashes[i] += 1
                    done[i] = bool(terms_l[i] or truncs_l[i])
                
                obs_dict = nxt_dict
                for i in range(self.num_agents):
                    self.prev_obs[i] = obs_list[i].copy()
                    self.prev_actions[i] = actions[i]
        
        env_wrapped.close()
        avg = [t / max(1, episodes) for t in totals]
        crash_rate = [c / max(1, episodes) for c in crashes]
        return avg, crash_rate


class HyperParams:
    """Placeholder class for hyperparameters."""
    pass
