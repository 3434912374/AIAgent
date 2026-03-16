import numpy as np
import gymnasium as gym
from gymnasium import spaces
from ..reward_funcs import pareto_reward

class ClusterEnv(gym.Env):
    def __init__(self, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes
        self.action_space = spaces.Discrete(self.num_nodes)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_nodes,), dtype=np.float32)
        self.state = np.zeros(self.num_nodes, dtype=np.float32)

    def step(self, action):
        # 兼容处理：SB3 有时会传 numpy 数组进来
        if isinstance(action, np.ndarray):
            action = int(action.item())
            
        load = self.state[action]
        reward = pareto_reward(load)
        
        # 模拟：被分配任务的节点负载增加，其他节点衰减
        self.state[action] = min(1.0, self.state[action] + 0.3)
        self.state = np.maximum(0.0, self.state - 0.05).astype(np.float32)
        
        # Gymnasium 规定必须返回 5 个值: obs, reward, terminated, truncated, info
        return self.state, float(reward), False, False, {}

    def reset(self, seed=None, options=None):
        # 修复 Gymnasium 新版对 options 参数的硬性要求
        super().reset(seed=seed)
        self.state = np.random.rand(self.num_nodes).astype(np.float32) * 0.5
        return self.state, {}