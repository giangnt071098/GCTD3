import csv 
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
parser = argparse.ArgumentParser()
parser.add_argument("--file_name", default="trajectory_file")
parser.add_argument("--agent", default = "TD3")
args = parser.parse_args()
filename = os.path.join(args.file_name + ".csv")
phase1, t1 = [], []
phase2, t2 = [], []
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
            array_reward.append(float(row[0]))
            timestep.append(i*100)
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
        max_step = 400 #len(list_csv)
        for i, row in enumerate(range(1, max_step, 1)):
            rows = list_csv[row][0].split('\t')
            hr.append(float(rows[4]))
            hl.append(float(rows[5]))
            kr.append(float(rows[8]))
            kl.append(float(rows[9]))
            ar.append(float(rows[12]))
            al.append(float(rows[13]))
            fr.append(float(rows[16]))
            fl.append(float(rows[17]))
            if float(rows[16]) + float(rows[17]) == 1:
                phase1.append(-float(rows[0]))
                t1.append(i)
            else:
                phase2.append(-float(rows[0]))
                t2.append(i)
            timestep.append(i)
    # for i in range(8):
    #     plt.figure(labels[i])
    #     plt.plot(timestep, gaussian_filter1d(array[i], sigma =3), label=labels[i])
    #     plt.xlabel('timestep')
    #     plt.ylabel(labels[i])
    #     plt.title('Walker robot')
    #     plt.legend()


    #plot to compare
    smooth = 4
    fig, implot = plt.subplots(1,3, figsize=(28,4))
    implot[0].plot(timestep, gaussian_filter1d(array[0], sigma =smooth), timestep, gaussian_filter1d(array[1], sigma =smooth))
    implot[1].plot(timestep, gaussian_filter1d(array[2], sigma =smooth), timestep, gaussian_filter1d(array[3], sigma = smooth))
    implot[2].plot(timestep, gaussian_filter1d(array[4], sigma =smooth), timestep, gaussian_filter1d(array[5], sigma = smooth))
    for i in range(len(implot)):

        implot[i].set_title(args.agent)
        implot[i].legend([labels[i*2], labels[i*2 +1]], loc = 'lower right')
    plt.setp(implot, xlabel = 'timestep')
    plt.setp(implot[0], ylabel = 'Rotation angle (radian)')

    g1, g2 = 4, 5
    plt.figure(labels[g1] + ' and ' + labels[g2])
    plt.plot(timestep, gaussian_filter1d(array[g1], sigma =3), timestep, gaussian_filter1d(array[g2], sigma =3))
    plt.xlabel('timestep')
    plt.ylabel('Rotation angle (radian)')
    plt.title(args.agent)
    plt.legend([labels[g1], labels[g2]], loc='lower right')

    plt.figure('z position')
    plt.plot(t1, gaussian_filter1d(phase1, sigma =5)) 
    plt.plot(t2, gaussian_filter1d(phase2, sigma =5))

    plt.show()
    print(np.mean(np.array(phase1)), np.mean(np.array(phase2)))

else:
    print("There is no file with that name!!!")
