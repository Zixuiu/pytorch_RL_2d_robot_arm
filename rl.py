import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time

#代码实现DDPG算法

device = 'cuda' if torch.cuda.is_available() else 'cpu'
directory='model/'

DDPG_CONFIG={
'LR_ACTOR':0.001,  # Actor学习率
'LR_CRITIC':0.001,  # Critic学习率
'GAMMA':0.9,  # 折扣因子
'TAU':0.01,  # 软更新参数
'MEMORY_CAPACITY':10000,  # 经验回放缓存大小
'BATCH_SIZE':32,  # 每次训练的样本数量

}

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)  # 第一层全连接层，输入维度为state_dim，输出维度为400
        self.l2 = nn.Linear(400, 300)  # 第二层全连接层，输入维度为400，输出维度为300
        self.l3 = nn.Linear(300, action_dim)  # 第三层全连接层，输入维度为300，输出维度为action_dim
        self.max_action = max_action  # 动作的最大值

    def forward(self, x):
        x = F.relu(self.l1(x))  # 使用ReLU激活函数进行非线性变换
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))  # 使用tanh激活函数将输出限制在[-max_action, max_action]范围内
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)  # 第一层全连接层，输入维度为state_dim+action_dim，输出维度为400
        self.l2 = nn.Linear(400 , 300)  # 第二层全连接层，输入维度为400，输出维度为300
        self.l3 = nn.Linear(300, 1)  # 第三层全连接层，输入维度为300，输出维度为1

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))  # 将状态和动作连接起来作为输入
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.memory = np.zeros((DDPG_CONFIG['MEMORY_CAPACITY'], state_dim * 2 + action_dim + 1), dtype=np.float32)  # 初始化经验回放缓存
        self.pointer = 0  # 经验回放缓存指针
        self.memory_full = False  # 经验回放缓存是否已满
        self.state_dim=state_dim  # 状态维度
        self.action_dim=action_dim  # 动作维度

        self.actor = Actor(state_dim, action_dim, max_action[1]).to(device)  # 创建Actor网络
        self.actor_target = Actor(state_dim, action_dim, max_action[1]).to(device)  # 创建目标Actor网络
        self.actor_target.load_state_dict(self.actor.state_dict())  # 将目标Actor网络的参数初始化为与Actor网络相同
        self.actor_optimizer = optim.Adam(self.actor.parameters(), DDPG_CONFIG['LR_ACTOR'])  # 创建Actor网络的优化器

        self.critic = Critic(state_dim, action_dim).to(device)  # 创建Critic网络
        self.critic_target = Critic(state_dim, action_dim).to(device)  # 创建目标Critic网络
        self.critic_target.load_state_dict(self.critic.state_dict())  # 将目标Critic网络的参数初始化为与Critic网络相同
        self.critic_optimizer = optim.Adam(self.critic.parameters(), DDPG_CONFIG['LR_CRITIC'])  # 创建Critic网络的优化器

    def choose_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)  # 将状态转换为张量并移动到指定设备
        return self.actor(state).cpu().data.numpy().flatten()  # 使用Actor网络选择动作，并将结果移动到CPU上并转换为numpy数组

    def store_transition(self, states, actions, rewards, states_next):
        transitions = np.hstack((states, actions, [rewards], states_next))  # 将状态、动作、奖励和下一个状态连接起来作为一条经验
        index = self.pointer % DDPG_CONFIG['MEMORY_CAPACITY']  # 计算经验回放缓存的索引
        self.memory[index, :] = transitions  # 将经验存储到经验回放缓存中
        self.pointer += 1  # 更新经验回放缓存指针
        if self.pointer > DDPG_CONFIG['MEMORY_CAPACITY']:      # 如果经验回放缓存已满
            self.memory_full = True  # 设置经验回放缓存已满的标志

    def learn(self):
        indices = np.random.choice(DDPG_CONFIG['MEMORY_CAPACITY'], size = DDPG_CONFIG['BATCH_SIZE'])  # 从经验回放缓存中随机选择一批样本
        bt = torch.Tensor(self.memory[indices, :])  # 将样本转换为张量
        state = bt[:, :self.state_dim].to(device)  # 获取状态
        action = bt[:, self.state_dim: self.state_dim + self.action_dim].to(device)  # 获取动作
        reward = bt[:, -self.state_dim - 1: -self.state_dim].to(device)  # 获取奖励
        next_state = bt[:, -self.state_dim:].to(device)  # 获取下一个状态

        # 计算目标Q值
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (DDPG_CONFIG['GAMMA'] * target_Q).detach()

        # 获取当前Q值估计
        current_Q = self.critic(state, action)

        # 计算Critic损失
        critic_loss = F.mse_loss(current_Q, target_Q)
        # 优化Critic网络
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 计算Actor损失
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # 优化Actor网络
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新目标网络的参数
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(DDPG_CONFIG['TAU'] * param.data + (1 - DDPG_CONFIG['TAU']) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(DDPG_CONFIG['TAU'] * param.data + (1 - DDPG_CONFIG['TAU']) * target_param.data)

    def save(self):
        torch.save(self.actor.state_dict(), directory + 'actor.pth')  # 保存Actor网络的参数
        torch.save(self.critic.state_dict(), directory + 'critic.pth')  # 保存Critic网络的参数
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def restore(self):
        self.actor.load_state_dict(torch.load(directory + 'actor.pth'))  # 加载Actor网络的参数
        self.critic.load_state_dict(torch.load(directory + 'critic.pth'))  # 加载Critic网络的参数
        print("====================================")
        print("model has been loaded...")
        print("====================================")
