"""
使其更加健壮。
当手指停在最终位置50步时停止。
特征和奖励工程。
"""
from env import ArmEnv  # 导入环境模块
from rl import DDPG  # 导入强化学习模块
import time as t  # 导入时间模块
import numpy as np  # 导入numpy模块
MAX_EPISODES = 500  # 最大训练轮数
MAX_EP_STEPS = 200  # 每轮最大步数
ON_TRAIN = False  # 是否训练模式


#该代码使用DDPG算法对一个机械臂控制环境进行训练和评估。训练过程中，机械臂根据当前状态选择动作，并观察下一个状态和奖励。
# 训练过程中，每个动作都会存储为经验，并在存储满后开始学习。评估过程中，加载已训练好的模型，并观察机械臂在环境中的表现。


# 设置环境
env = ArmEnv()
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound

# 设置强化学习方法（连续动作）
rl = DDPG(s_dim,a_dim, a_bound)

steps = []
def train():
    # 开始训练
    for i in range(MAX_EPISODES):
        s = env.reset()  # 重置环境
        ep_r = 0.  # 初始化总奖励为0
        for j in range(MAX_EP_STEPS):
            env.render()  # 渲染环境
            #a=env.sample_action()
            a = rl.choose_action(s)  # 根据当前状态选择动作
            if(np.isnan(a[0])):
                a[0]=0
            if(np.isnan(a[1])):
                a[1]=0
            s_, r, done = env.step(a)  # 执行动作并观察下一个状态和奖励
            rl.store_transition(s, a, r, s_)  # 存储经验

            ep_r += r  # 更新总奖励
            if rl.memory_full:
                # 存储满了后开始学习
                rl.learn()

            s = s_  # 更新当前状态
            if done or j == MAX_EP_STEPS-1:
                print('Ep: %i | %s | ep_r: %.1f | step: %i' % (i, '---' if not done else 'done', ep_r, j))
                break
    rl.save()  # 保存模型


def eval():
    rl.restore()  # 加载模型
    env.render()  # 渲染环境
    env.viewer.set_vsync(True)
    while True:
        s = env.reset()  # 重置环境
        for _ in range(200):
            env.render()  # 渲染环境
            a = rl.choose_action(s)  # 根据当前状态选择动作
            s, r, done = env.step(a)  # 执行动作并观察下一个状态和奖励
            if done:
                break


if ON_TRAIN:
    train()  # 进行训练
else:
    eval()  # 进行评估
