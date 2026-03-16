from stable_baselines3 import PPO
from .envs.cluster_env import ClusterEnv

class PPOScheduler:
    def __init__(self, num_nodes, timesteps):
        self.env = ClusterEnv(num_nodes)
        print(f"[RL] 正在预训练 PPO 调度器 (步数: {timesteps})...")
        # 添加了 device="cpu" 以消除警告并提升模拟训练速度
        self.model = PPO("MlpPolicy", self.env, verbose=0, device="cuda")
        self.model.learn(total_timesteps=timesteps)
        self.obs, _ = self.env.reset()

    def get_best_node(self):
        # 预测时强制获取标量动作
        action, _ = self.model.predict(self.obs, deterministic=True)
        self.obs, _, _, _, _ = self.env.step(action)
        return int(action)