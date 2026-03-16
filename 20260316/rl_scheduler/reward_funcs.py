# 多目标 Pareto 奖励

def pareto_reward(node_load,max_load_threshold=0.8):
    if node_load>max_load_threshold:
        return -10.0  # 超过负载阈值，给予较大负奖励
    return 1.0-node_load#负载越低奖励越高，鼓励分散任务