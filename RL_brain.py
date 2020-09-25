"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list (range(行动数量)) [0,1,2,3]
        self.lr = learning_rate #学习效率
        self.gamma = reward_decay #奖励衰减
        self.epsilon = e_greedy #贪婪指数
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64) #q_table(红框的坐标数量作为列名)

    def choose_action(self, observation):#字符串格式的observation
        self.check_state_exist(observation) #observation:红色方框坐标，判断是否存在，不存在添加

        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions) #index值
        return action

    def learn(self, s, a, r, s_):#str(observation), action, reward, str(observation_)
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a] #预计移动方向
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
            #self.q_table.loc[s_, :].max()：最大的值
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  #  更新对应的 state-action 值

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )