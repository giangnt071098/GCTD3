import csv 
import matplotlib.pyplot as plt
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("--file_name", default="reward_file")
parser.add_argument("--policy", default = "TD3")
args = parser.parse_args()
filename = args.file_name + ".csv"
if args.policy == 'DDPG':
    filename = os.path.join('results', 'DDPG', filename)

array_reward = []
array = []
timestep = []
hr, hl, kr, kl, ar, al, fr, fl = [], [], [], [], [], [], [], []
array = [hr, hl, kr, kl, ar, al, fr,fl]
labels = ['right hip', 'left hip', 'right knee', ' left knee', 'right ankle', 'left ankle', 'right foot contact', 'left foot contact']
if args.file_name == "reward_file":
    with open(filename,newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for i, row in enumerate(list(spamreader)[1:]):
            if i%2 ==0:
                array_reward.append(float(row[0]))
                timestep.append(i*50)
    plt.figure("reward")
    plt.plot(timestep,array_reward, label="reward")
    plt.xlabel('timestep')
    plt.ylabel("reward")
    plt.title('Walker robot')
    plt.legend()
    plt.show()
elif args.file_name == "trajectory_file":
    with open(filename, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        list_csv = list(spamreader)
        for i, row in enumerate(range(1,len(list_csv))):
            rows = list_csv[row][0].split('\t')
            hr.append(float(rows[4]))
            hl.append(float(rows[5]))
            kr.append(float(rows[8]))
            kl.append(float(rows[9]))
            ar.append(float(rows[12]))
            al.append(float(rows[13]))
            fr.append(float(rows[16]))
            fl.append(float(rows[17]))
            timestep.append(i)
    for i in range(8):
        plt.figure(labels[i])
        plt.plot(timestep,array[i], label=labels[i])
        plt.xlabel('timestep')
        plt.ylabel(labels[i])
        plt.title('Walker robot')
        plt.legend()
        plt.show()
else:
    print("There is no file with that name!!!")
