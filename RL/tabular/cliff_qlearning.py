import gymnasium as gym
import numpy as np

class QLearningAgent:
    def __init__(self, obs_n, act_n, learning_rate=0.01, gamma=0.9, e_greedy=0.1):
        self.act_n = act_n
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = e_greedy
        self.Q = np.zeros((obs_n, act_n))

    def sample(self, obs):
        if np.random.uniform(0, 1) < (1.0 - self.epsilon):
            action = self.predict(obs)
        else:
            action = np.random.choice(self.act_n)
        return action

    def predict(self, obs):
        Q_list = self.Q[obs, :]
        maxQ = np.max(Q_list)
        action_list = np.where(Q_list == maxQ)[0]
        action = np.random.choice(action_list)
        return action
    
    def learn(self, obs, action, reward, next_obs, next_action, done):
        predict_Q = self.Q[obs, action]
        if done:
            target_Q = reward
        else:
            target_Q = reward + self.gamma * np.max(self.Q[next_obs, :])
        self.Q[obs, action] += self.lr * (target_Q - predict_Q)

    def save(self):
        npy_file = './q_table.npy'
        np.save(npy_file, self.Q)
        print('save q_table to:', npy_file)
    
    def restore(self, npy_file='./q_table.npy'):
        self.Q = np.load(npy_file)
        print('restore q_table from:', npy_file)

def run_episode(env, agent, render=False):
    total_step = 0
    total_reward = 0
    obs, info = env.reset()  # ✅ 解包
    action = agent.sample(obs)
    while True:
        next_obs, reward, terminated, truncated, info = env.step(action)  # ✅ 5元组
        done = terminated or truncated
        next_action = agent.sample(next_obs)
        agent.learn(obs, action, reward, next_obs, next_action, done)  # ✅ 传 next_action 和 done
        action = next_action
        obs = next_obs
        total_reward += reward
        total_step += 1
        # if render:
        #     env.render()
        if done:
            break
    return total_reward, total_step

def test_episode(agent, env, render=False):
    total_reward = 0
    obs, info = env.reset()
    while True:
        action = agent.predict(obs)  # ✅ 测试时用 greedy（不随机）
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        if render:
            env.render()
        if done:
            break
        obs = next_obs
    return total_reward

# 创建环境
env = gym.make('CliffWalking-v1')

# 初始化 agent
agent = QLearningAgent(
    obs_n=env.observation_space.n,
    act_n=env.action_space.n,
    learning_rate=0.1,
    gamma=0.9,
    e_greedy=0.1
)

# 训练
for episode in range(5000):
    ep_reward, ep_steps = run_episode(env, agent, render=True)
    if episode % 10 == 0:
        print(f'Episode: {episode}, Reward: {ep_reward}, Steps: {ep_steps}')

# 测试（使用确定性策略）
test_env = gym.make('CliffWalking-v1', render_mode='human')  # 可视化测试
test_reward = test_episode(agent, test_env, render=True)
print('Test reward:', test_reward)
# test_env.close()
env.close()