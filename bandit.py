import numpy as np


class Epsilon_Greedy_Bandit:
    def __init__(self):
        self.arm_values = np.random.normal(0, 1, 4)  # arm_values 各臂收益的真实值
        self.K = np.zeros(4)  # K=4 个臂
        self.est_values = np.zeros(4)  # est_values 各臂收益初始估计值 0

    # def get_reward(self,action):
    #     noise = np.random.normal(0,0.1)                 #给获取的汇报加入噪声
    #     reward = self.arm_values[action]+noise
    #     return reward

    def choose_eps_greedy(self, epsilon):  # choose_eps_greedy 选择动作
        rand_num = np.random.random()
        print(rand_num)
        if epsilon > rand_num:
            return np.random.randint(4)
        else:
            return np.argmax(self.est_values)

    def update_est(self, action, reward):  # update_est 更新所选臂的估计收益
        self.K[action] += 1
        print("action:", action, "k[action]:", self.K[action])
        alpha = 1. / self.K[action]
        self.est_values[action] += alpha * (reward - self.est_values[action])

class UCB_Bandit:
    def __init__(self):
        self.arm_values = np.random.normal(0, 1, 4)  # arm_values 各臂收益的真实值
        self.K = np.zeros(4)  # K=4 个臂
        self.est_values = np.zeros(4)  # est_values 各臂收益初始估计值 0
        self.choose_count = np.zeros(4)  # 某个臂被选择次数

    # def get_reward(self,action):
    #     noise = np.random.normal(0,0.1)                 #给获取的回报加入噪声
    #     reward = self.arm_values[action]+noise
    #     return reward

    def cal_delta(self, T, action):  # T rounds,
        c = 0.5
        if self.choose_count[action] == 0:
            return 1
        else:
            return np.sqrt(c * np.log(T) / self.choose_count[action])

    def choose_action(self, T):  # 选择动作
        upper_bound_probs = [self.est_values[action] + self.cal_delta(T, action) for action in range(4)]
        print("upb:", upper_bound_probs)
        action = np.argmax(upper_bound_probs)
        return action

    def update_est(self, action, reward):  # update_est 更新所选臂的估计收益
        self.K[action] += 1
        print("action:", action, "k[action]:", self.K[action])
        alpha = 1. / self.K[action]
        self.est_values[action] += alpha * (reward - self.est_values[action])