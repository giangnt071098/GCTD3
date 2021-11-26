import csv 
import matplotlib.pyplot as plt
import os
from scipy.ndimage.filters import gaussian_filter1d
import pandas as pd

reward_file = "reward_file.csv"
DDPG_reward = os.path.join('DDPG', reward_file)
TD3_reward = os.path.join('TD3_gauss_noise', reward_file)
SAC_reward = os.path.join('SAC', reward_file)
als_reward = [TD3_reward, DDPG_reward, SAC_reward]

name_als = ['TD3', 'DDPG', 'SAC']
color = ['#FF0000','#00ffff','#00ff0c']
df = []
for i in range(3):
    #df[name_als[i]] = pd.read_csv(als_reward[i])
    df.append(pd.read_csv(als_reward[i]))

with plt.style.context('Solarize_Light2'):
    for (wd, c) in zip(df, color):
        s = gaussian_filter1d(wd.avg_reward, sigma=4)
        plt.plot(wd.index*100, s ,linestyle='--',color=c) 

    
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.xlabel('Timestep', fontsize=14)
    plt.ylabel('Reward', fontsize=14)
    fig.suptitle('Average reward', fontsize=16)

    for (wd, c) in zip(df, color):
        plt.plot(wd.index*100, wd.avg_reward, color = c, linewidth=0.5)

    plt.legend(name_als)
plt.show()
