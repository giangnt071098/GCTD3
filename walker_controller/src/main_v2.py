from os import stat

import torch
from GCTD3 import GCTD3
from GCActorTD3 import GCActorTD3
from GCActorTD3his import GCActorTD3his
from TD3 import TD3
from GCTD3his import GCTD3his
from buffer import ReplayBuffer
import utils


import gc
gc.enable()


import roslib
import rospy
import rostopic
import random
import time
import math
import csv
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelConfiguration

from control_msgs.msg import JointControllerState
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import LinkStates
from gazebo_msgs.msg import ContactsState
from std_msgs.msg import Float64
from std_msgs.msg import String
from sensor_msgs.msg import Joy

import threading
from scipy.interpolate import interp1d
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--expl_noise", default=0.1, type=float)
parser.add_argument("--load_model", action="store_true") 
parser.add_argument("--save_model", action="store_true") 
parser.add_argument("--file_name", default="GCTD3_Walking_Robot")
parser.add_argument("--batch_size", default=32, type = int)
parser.add_argument("--agent", default="GCTD3")
args = parser.parse_args()

FILE_NAME = args.file_name
EPISODES = 500000
TEST = 5
BUFFER_SIZE = 1000000
START_TRAINING = 10000#args.batch_size*10 #10000
NUMBER_HISTORY = 5
torch.manual_seed(2234)
np.random.seed(55)

reward_file = "reward_file.csv"
trajectory_file = "trajectory_file.csv"


pubHipR = rospy.Publisher('/waist_thighR_position_controller/command', Float64, queue_size=10)
pubHipL = rospy.Publisher('/waist_thighL_position_controller/command', Float64, queue_size=10)
pubKneeR = rospy.Publisher('/thighR_shankR_position_controller/command', Float64, queue_size=10)
pubKneeL = rospy.Publisher('/thighL_shankL_position_controller/command', Float64, queue_size=10)
pubAnkelR = rospy.Publisher('/shankR_footR_position_controller/command', Float64, queue_size=10)
pubAnkelL = rospy.Publisher('/shankL_footL_position_controller/command', Float64, queue_size=10)
reset_simulation = rospy.ServiceProxy('/gazebo/reset_world', Empty)
reset_joints = rospy.ServiceProxy('/gazebo/set_model_configuration', SetModelConfiguration)
unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)

rospy.init_node('walker_control_script')



env = [(8,2), 6] # [state_dim, action_dim] (8 is number of nodes, 2 is number of feature)
parent_child = ((1,0),(2,0),(1,2), (2,1),(3,1),(4,2),(5,3),(6,4)) # delete (2,1)
Adj_matrix = utils.get_adjacency_matrix(env[0][0], parent_child)
replay_buffer = ReplayBuffer((env[0][0], env[0][1]*NUMBER_HISTORY), env[1], max_size=BUFFER_SIZE)

if args.agent == "GCTD3":
    agent = GCTD3(env[0][0], env[0][1]*NUMBER_HISTORY, env[1], Adj_matrix)
    FILE_NAME = "GCTD3_Walking_Robot"
    PATH_LOAD = FILE_NAME
    freq = 50 # 18, 0.055s
if args.agent == "GCActorTD3":
    agent = GCActorTD3(env[0][0], env[0][1], env[1], Adj_matrix)
    FILE_NAME = "GCActorTD3_Walking_Robot"
    PATH_LOAD = "GCActorTD3v2_3/GCActorTD3_Walking_Robot_final"
    freq = 50 #32
if args.agent == "GCActorTD3his":
    agent = GCActorTD3his(env[0][0], env[0][1]*NUMBER_HISTORY, env[1], Adj_matrix)
    FILE_NAME = "GCActorTD3his_Walking_Robot"
    PATH_LOAD = FILE_NAME
    freq = 50 #32
if args.agent == "GCTD3his":
    agent = GCTD3his(env[0][0], env[0][1]*NUMBER_HISTORY, env[1], Adj_matrix)
    FILE_NAME = "GCTD3hisv2_Walking_Robot"
    PATH_LOAD = "GCTD3his_Walking_Robot_520.26"
    freq = 50
if args.agent == "TD3":
    agent = TD3(env[0][0]*env[0][1], env[1])
    FILE_NAME = "TD3_Walking_Robot"
    freq = 50
# load and save model
replaybuffer_mem = "replaybuffer_{}.npy".format(args.agent)
save = False
if args.load_model == True:
    agent.load_model(PATH_LOAD)
    agent.training = True
    # load_mem = np.load(replaybuffer_mem,  allow_pickle=True)
    # for mdp in load_mem:
    #     replay_buffer.add(*mdp)
    print("Buffer Memmory: ", replay_buffer.size)
if args.save_model == True:
    save = True

rate = rospy.Rate(freq) 
class RobotState(object):
    def __init__(self):
        self.waist_z = 0.0
        self.waist_y = 0.0
        self.outer_ring_inner_ring_theta = 0.0
        self.hipr_theta = 0.0
        self.hipr_theta_dot = 0.0
        self.hipl_theta = 0.0
        self.hipl_theta_dot = 0.0
        self.kneer_theta = 0.0
        self.kneer_theta_dot = 0.0
        self.kneel_theta = 0.0
        self.kneel_theta_dot = 0.0
        self.ankelr_theta = 0.0
        self.ankelr_theta_dot = 0.0
        self.ankell_theta = 0.0
        self.ankell_theta_dot = 0.0
        self.vel_y = 0.0
        self.vel_z = 0.0
        self.footr_contact = 0
        self.footl_contact = 0
        self.robot_state = [[self.waist_z, self.vel_z], [self.hipr_theta, self.hipr_theta_dot], [self.hipl_theta, self.hipl_theta_dot], \
        [self.kneer_theta, self.kneer_theta_dot], [self.kneel_theta, self.kneel_theta_dot], [self.ankelr_theta, self.ankelr_theta_dot], \
        [self.ankell_theta, self.ankell_theta_dot], [self.footr_contact, self.footl_contact]]
        
        self.latest_reward = 0.0
        self.best_reward = -1e4
        self.episode = 0
        self.last_outer_ring_inner_ring_theta = 0.0
        self.last_time = 0.0

        self.fall = 0
        self.done = False
        self.avg_reward = 0.0
        self.count_state = 0
        self.count_contactL = 0
        self.count_contactR = 0
        self.count_hipL = 0
        self.count_hipR = 0
        self.signhipR =1
        self.signhipL =1
        self.last_y = 0

class Publisher(threading.Thread):
    def __init__(self, pubHipR, pubHipL, pubKneeR, pubKneeL, pubAnkelR, pubAnkelL, rate):
        threading.Thread.__init__(self)
        self.counter = 0
        self.pubHipR = pubHipR
        self.pubHipL = pubHipL
        self.pubKneeR = pubKneeR
        self.pubKneeL = pubKneeL
        self.pubAnkelR = pubAnkelR
        self.pubAnkelL = pubAnkelL
        self.rate = rate


    def run(self):
        publisher(self.pubHipR, self.pubHipL, self.pubKneeR, self.pubKneeL, self.pubAnkelR, self.pubAnkelL, self.rate, self.counter)

robot_state = RobotState()


def reset():
    # ['waist_thighR', 'waist_thighL', 'thighR_shankR', 'thighL_shankL', 'outer_ring_inner_ring', 'inner_ring_boom', 'boom_waist']
    rospy.wait_for_service('gazebo/reset_world')
    try:
        reset_simulation()
    except(rospy.ServiceException) as e:
        print("reset_world failed!")


    rospy.wait_for_service('gazebo/set_model_configuration')

    try:
        reset_joints("model_walking_robot", "robot_description", ['boom_waist', 'outer_inner', 'shankL_footL', 'shankR_footR', 'thighL_shankL', 'thighR_shankR', 'waist_thighL', 'waist_thighR'], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        robot_state.last_outer_ring_inner_ring_theta = 0.0
    except (rospy.ServiceException) as e:
        print("reset_joints failed!")

    rospy.wait_for_service('/gazebo/pause_physics')
    try:
        pause()
    except (rospy.ServiceException) as e:
        print("rospause failed!'")

    set_robot_state()

    # print "called reset()"



def set_robot_state():
    robot_state.robot_state = [[robot_state.waist_z, robot_state.vel_z], [robot_state.hipr_theta, robot_state.hipr_theta_dot], [robot_state.hipl_theta, robot_state.hipl_theta_dot], \
        [robot_state.kneer_theta, robot_state.kneer_theta_dot], [robot_state.kneel_theta, robot_state.kneel_theta_dot], [robot_state.ankelr_theta, robot_state.ankelr_theta_dot], \
        [robot_state.ankell_theta, robot_state.ankell_theta_dot], [robot_state.footr_contact, robot_state.footl_contact]]
def take_action(action, observes, y_set):
    rospy.wait_for_service('/gazebo/unpause_physics')

    try:
        unpause()
    except (rospy.ServiceException) as e:
        print("/gazebo/pause_physics service call failed")

    pubHipR.publish(action[0])
    pubKneeR.publish(action[1])
    pubAnkelR.publish(action[2])
    pubHipL.publish(action[3])
    pubKneeL.publish(action[4])
    pubAnkelL.publish(action[5])

    # constant weight 
    ctrl_cost_weights = 0.0
    cost_fall_weights = 0.005#3.0
    constant_vy = 0.9
    c1,c2 = 0.0, 1.0
    sum_vel = c1*np.abs(robot_state.hipl_theta_dot**2 + robot_state.hipr_theta_dot**2)
    height = observes[0][::2]
    kneer_theta = observes[3][::2]
    kneel_theta = observes[4][::2]
    mean_h = np.mean(height)
    expl_noise = args.expl_noise
    delta = y_set[0] - y_set[-1]
    

    reward = -0.1
    reward += min(robot_state.vel_y, 0.9) + 0.2*delta
    
    # if delta >0 :
    #     reward += 22*delta
    # else:
    #     reward -= delta
    #reward += c2*min(robot_state.vel_y, constant_vy) + ctrl_cost_weights*sum_vel- cost_fall_weights*np.abs(np.abs(robot_state.waist_z -mean_h) - 0.002)
    #reward += robot_state.vel_y - ctrl_cost_weights*np.sum(np.square(action)) - cost_fall_weights*np.abs(robot_state.waist_z -0.017)
    #reward += constant_vy - np.abs(constant_vy - robot_state.vel_y) - ctrl_cost_weights*np.sum(np.square(action)) - cost_fall_weights*(robot_state.waist_z -0.016)**2


    # --- 2 states of walking with 2 other target_h
    for col in range(0, observes.shape[1], 2):
        state = observes[:,col:col+2]
        if sum(state[7]) == 0:
            reward -= 0.01
        else:
            if sum(state[7]) == 1:
                target_h = 0.001
            else:
                target_h = 0.025
            reward -= cost_fall_weights*np.abs(robot_state.waist_z - target_h)

    # if (np.max(kneer_theta)-np.min(kneer_theta))<0.1:
    #     reward -= 0.001
    # if (np.max(kneel_theta)-np.min(kneel_theta))<0.1:
    #     reward -= 0.001


    if (robot_state.hipl_theta) * robot_state.signhipL > 0:
        robot_state.count_hipL += 1
    else:
        robot_state.count_hipL = 0
        robot_state.signhipL *= -1
    if (robot_state.hipr_theta)*robot_state.signhipR > 0:
        robot_state.count_hipR += 1
    else:
        robot_state.count_hipR = 0
        robot_state.signhipR *= -1

    # if robot_state.waist_z < -0.012:
    #     reward += 0.01 * robot_state.waist_z #punish if robot stand on tiptoe 
    # if np.abs(robot_state.waist_y - robot_state.last_y) <1e-3:
    #     robot_state.count_state += 1
    #     reward -= 0.001*(robot_state.count_state)
    # else:
    #     robot_state.count_state = 0
    #     robot_state.last_y = robot_state.waist_y

    if robot_state.footl_contact:
        robot_state.count_contactL += 1
    else:
        robot_state.count_contactL = 0

    if robot_state.footr_contact:
        robot_state.count_contactR += 1
    else:
        robot_state.count_contactR = 0
    #print(observes)
    

    #### reset environment
    if robot_state.count_contactR > 40 or robot_state.count_contactL > 40:
        robot_state.done = True
        robot_state.fall = 1
        robot_state.count_contactL, robot_state.count_contactR = 0, 0
        robot_state.count_hipR, robot_state.count_hipL =0, 0
        expl_noise = 0.3
        reward -= 0.2
    if robot_state.count_hipL>50 or robot_state.count_hipR > 50:
        robot_state.done = True
        robot_state.fall = 1
        robot_state.count_hipR, robot_state.count_hipL =0, 0
        reward -= 0.5

    # if robot_state.count_state > 30:
    #     robot_state.done = True
    #     robot_state.fall = 1
    #     robot_state.count_state =0
    #     expl_noise = 0.3
    
    if robot_state.waist_z > 0.1:
        reward -= 10
        robot_state.done = True
        robot_state.fall = 1
    if robot_state.outer_ring_inner_ring_theta > 14.0:
        reward += 10
        robot_state.done = True
        robot_state.fall = 1
        #print("REACH TO THE END!")
    rate.sleep()
    return reward, robot_state.done, expl_noise

def take_action_ver2(action, observes):
    rospy.wait_for_service('/gazebo/unpause_physics')

    try:
        unpause()
    except (rospy.ServiceException) as e:
        print("/gazebo/pause_physics service call failed")

    pubHipR.publish(action[0])
    pubKneeR.publish(action[1])
    pubAnkelR.publish(action[2])
    pubHipL.publish(action[3])
    pubKneeL.publish(action[4])
    pubAnkelL.publish(action[5])

    # constant weight 
    ctrl_cost_weights = 0.0
    cost_fall_weights = 0.005#3.0
    constant_vy = 0.9
    c1,c2 = 0.0, 1.0
    sum_vel = c1*np.abs(robot_state.hipl_theta_dot**2 + robot_state.hipr_theta_dot**2)
    height = observes[0][::2]
    kneer_theta = observes[3][::2]
    kneel_theta = observes[4][::2]
    mean_h = np.mean(height)
    expl_noise = args.expl_noise

    

    reward = -0.1
    # reward += min(robot_state.vel_y, 0.85) + 1*(robot_state.waist_y-robot_state.last_y)
    reward += 10*(robot_state.waist_y - robot_state.last_y)
    #reward += c2*min(robot_state.vel_y, constant_vy) + ctrl_cost_weights*sum_vel- cost_fall_weights*np.abs(np.abs(robot_state.waist_z -mean_h) - 0.002)
    #reward += robot_state.vel_y - ctrl_cost_weights*np.sum(np.square(action)) - cost_fall_weights*np.abs(robot_state.waist_z -0.017)
    #reward += constant_vy - np.abs(constant_vy - robot_state.vel_y) - ctrl_cost_weights*np.sum(np.square(action)) - cost_fall_weights*(robot_state.waist_z -0.016)**2


    # --- 2 states of walking with 2 other target_h
    for col in range(0, observes.shape[1], 2):
        state = observes[:,col:col+2]
        if sum(state[7]) == 0:
            reward -= 0.01
        else:
            if sum(state[7]) == 1:
                target_h = 0.001
            else:
                target_h = 0.025
            reward -= cost_fall_weights*np.abs(robot_state.waist_z - target_h)

    # if (np.max(kneer_theta)-np.min(kneer_theta))<0.1:
    #     reward -= 0.001
    # if (np.max(kneel_theta)-np.min(kneel_theta))<0.1:
    #     reward -= 0.001


    if (robot_state.hipl_theta) * robot_state.signhipL > 0:
        robot_state.count_hipL += 1
    else:
        robot_state.count_hipL = 0
        robot_state.signhipL *= -1
    if (robot_state.hipr_theta)*robot_state.signhipR > 0:
        robot_state.count_hipR += 1
    else:
        robot_state.count_hipR = 0
        robot_state.signhipR *= -1

    # if robot_state.waist_z < -0.012:
    #     reward += 0.01 * robot_state.waist_z #punish if robot stand on tiptoe 
    # if np.abs(robot_state.waist_y - robot_state.last_y) <1e-3:
    #     robot_state.count_state += 1
    #     reward -= 0.001*(robot_state.count_state)
    # else:
    #     robot_state.count_state = 0
    #     robot_state.last_y = robot_state.waist_y

    if robot_state.footl_contact:
        robot_state.count_contactL += 1
    else:
        robot_state.count_contactL = 0

    if robot_state.footr_contact:
        robot_state.count_contactR += 1
    else:
        robot_state.count_contactR = 0
    #print(observes)
    

    #### reset environment
    if robot_state.count_contactR > 40 or robot_state.count_contactL > 40:
        robot_state.done = True
        robot_state.fall = 1
        robot_state.count_contactL, robot_state.count_contactR = 0, 0
        robot_state.count_hipR, robot_state.count_hipL =0, 0
        expl_noise = 0.3
        reward -= 0.2
    if robot_state.count_hipL>50 or robot_state.count_hipR > 50:
        robot_state.done = True
        robot_state.fall = 1
        robot_state.count_hipR, robot_state.count_hipL =0, 0
        reward -= 0.5

    # if robot_state.count_state > 30:
    #     robot_state.done = True
    #     robot_state.fall = 1
    #     robot_state.count_state =0
    #     expl_noise = 0.3
    
    if robot_state.waist_z > 0.1:
        reward -= 10
        robot_state.done = True
        robot_state.fall = 1
    if robot_state.outer_ring_inner_ring_theta > 14.0:
        reward += 10
        robot_state.done = True
        robot_state.fall = 1
        #print("REACH TO THE END!")
    robot_state.last_y = robot_state.waist_y
    rate.sleep()
    return reward, robot_state.done, expl_noise

def callbackJointStates(data):
    # ['boom_waist', 'outer_ring_inner_ring', 'thighL_shankL', 'thighR_shankR', 'waist_thighL', 'waist_thighR']
    # if vel == 0 ['waist_thighR', 'waist_thighL', 'thighR_shankR', 'thighL_shankL', 'outer_ring_inner_ring', 'boom_waist']
    robot_state.data = data

    if len(data.velocity)!=0:

        robot_state.vel_z = data.velocity[0]
        robot_state.vel_y = data.velocity[1]
        robot_state.ankell_theta_dot = data.velocity[2]
        robot_state.ankelr_theta_dot = data.velocity[3]
        robot_state.kneel_theta_dot = data.velocity[4]
        robot_state.kneer_theta_dot = data.velocity[5]
        robot_state.hipl_theta_dot = data.velocity[6]
        robot_state.hipr_theta_dot = data.velocity[7]

        robot_state.waist_z = data.position[0] - 0.0235 #init distance between boom and waist = 0.023 (rostopic echo /joint_states)
        robot_state.waist_y = data.position[1]
        robot_state.outer_ring_inner_ring_theta = data.position[1]
        robot_state.ankell_theta = data.position[2]
        robot_state.ankelr_theta = data.position[3]
        robot_state.kneel_theta = data.position[4]
        robot_state.kneer_theta = data.position[5]
        robot_state.hipl_theta = data.position[6]
        robot_state.hipr_theta = data.position[7]
    else:
        #robot_state.__init__()
        robot_state.vel_z = 0
        robot_state.vel_y = 0
        robot_state.ankell_theta_dot = 0
        robot_state.ankelr_theta_dot = 0
        robot_state.kneel_theta_dot = 0
        robot_state.kneer_theta_dot = 0
        robot_state.hipl_theta_dot = 0
        robot_state.hipr_theta_dot = 0

        robot_state.waist_z = 0
        robot_state.waist_y = 0
        robot_state.outer_ring_inner_ring_theta = 0
        robot_state.ankell_theta = 0
        robot_state.ankelr_theta = 0
        robot_state.kneel_theta = 0
        robot_state.kneer_theta = 0
        robot_state.hipl_theta = 0
        robot_state.hipr_theta = 0

    set_robot_state()
    # rate.sleep()


def callbackSub(data):
    set_robot_state()

def callbackContactFootR(data):
    if not data.states:
        robot_state.footr_contact = 0
    else:
        robot_state.footr_contact = 1


def callbackContactFootL(data):
    if not data.states:
        robot_state.footl_contact = 0
    else:
        robot_state.footl_contact = 1

def listener():
    print("listener")

    rospy.Subscriber("/joint_states", JointState, callbackJointStates)
    rospy.Subscriber("/footR_contact_sensor_state", ContactsState, callbackContactFootR)
    rospy.Subscriber("/footL_contact_sensor_state", ContactsState, callbackContactFootL)





def publisher(pubHipR, pubHipL, pubKneeR, pubKneeL, pubAnkelR, pubAnkelL, rate, counter):

    while not rospy.is_shutdown():
        try:
            # writing rewards in the csv file
            file = open(reward_file, 'wt')
            writer = csv.writer(file)
            writer.writerow(['avg_reward'])

            
            number_step = 0
            robot_state.best_reward =200 # manual 
            last_action = np.zeros(env[1])
            expl_noise = args.expl_noise
            # training and testing
            for episode in range(EPISODES):
                reset()
                state = robot_state.robot_state
                observes = np.concatenate([state]*NUMBER_HISTORY, axis=1)
                y_set = np.zeros(NUMBER_HISTORY)
                # Train
                for steps in range(1600):
                    number_step += 1
                    action = agent.select_action(observes, expl_noise)
                    #action = [1, -0.5, -1, -0.5,-0.5,0]
                    #act = utils.modify_action(action)
                    # reward,done, expl_noise = take_action_ver2(action, observes)
                    reward, done, expl_noise = take_action(action, observes, y_set)
                    next_state = robot_state.robot_state
                    next_observes = np.append(next_state, observes[:,:env[0][1]*(NUMBER_HISTORY-1)], axis=1)
                    if np.abs(np.sum(np.array(action)- np.array(last_action)))>0.05:
                        replay_buffer.add(observes, action, next_observes, reward, done)
                    else:
                        expl_noise = 0.2
                    #replay_buffer.add(*utils.flip_transition(state, action, next_state, reward, done))
                    if replay_buffer.size > START_TRAINING:
                        agent.train(replay_buffer, batch_size = args.batch_size)
                    last_action = action
                    
                    observes = next_observes
                    y_set = np.append(robot_state.waist_y, y_set[:NUMBER_HISTORY-1])
                    if done:
                        robot_state.done = False
                        break
                robot_state.episode = episode
                # if replay_buffer.size > START_TRAINING:
                #     for _ in range(min(steps, 200)):
                #         agent.train(replay_buffer, batch_size= args.batch_size)
                # Testing:
                if episode % 50 == 0 and episode >= 50 and number_step >START_TRAINING:
                    traj_file = open(trajectory_file, 'wt')
                    traj_writer = csv.writer(traj_file, delimiter='\t')
                    traj_writer.writerow(['z', 'y', 'vel_z', 'vel_y', 'hr', 'hl', 'hr_dot', 'hl_dot', 'kr', 'kl', 'kr_dot', 'kl_dot', 'ar', 'al', 'ar_dot', 'al_dot', 'fr', 'fl'])

                    print("testing")
                    total_reward = 0
                    for i in range(TEST):
                        reset()
                        state = robot_state.robot_state
                        observes = np.concatenate([state]*NUMBER_HISTORY, axis=1)
                        y_set = np.zeros(NUMBER_HISTORY)
                        for steps in range(1600):
                            action = agent.predict_action(observes) # direct action for test
                            #act = utils.modify_action(action)
                            # reward,done, _ = take_action_ver2(action, observes)
                            reward,done, _ = take_action(action, observes, y_set)
                            #time.sleep(0.001) # sleep time code
                            state = robot_state.robot_state
                            observes = np.append(state, observes[:,:env[0][1]*(NUMBER_HISTORY-1)], axis=1)
                            traj_writer.writerow([robot_state.waist_z, robot_state.waist_y, robot_state.vel_z, robot_state.vel_y, robot_state.hipr_theta, robot_state.hipl_theta, robot_state.hipr_theta_dot, robot_state.hipl_theta_dot, \
                                                robot_state.kneer_theta, robot_state.kneel_theta, robot_state.kneer_theta_dot, robot_state.kneel_theta_dot, robot_state.ankelr_theta, robot_state.ankell_theta, robot_state.ankelr_theta_dot, \
                                                robot_state.ankell_theta_dot, robot_state.footr_contact, robot_state.footl_contact])
                            traj_file.flush()
                            y_set = np.append(robot_state.waist_y, y_set[:NUMBER_HISTORY-1])
                            total_reward += reward
                            if done:
                                robot_state.done = False
                                break
                    ave_reward = total_reward/TEST
                    robot_state.latest_reward = ave_reward
                    if ave_reward > robot_state.best_reward:
                        robot_state.best_reward = ave_reward
                        if save: agent.save_model(FILE_NAME)
                    if save and number_step > START_TRAINING and ave_reward > 400 and episode >1000: 
                        agent.save_model(FILE_NAME+"_"+str(round(ave_reward,2)))
                    if save and number_step > START_TRAINING and ave_reward > 300: 
                        agent.save_model(FILE_NAME+"_final")
                    robot_state.avg_reward = ave_reward
                    writer.writerow([ave_reward])
                    file.flush()

                    print("episode: ",episode, "number step: ", number_step,"Evaluation Average Reward: ",ave_reward)
                    print("best_reward: ", robot_state.best_reward)
                    print("Actor loss = ", agent.actor_loss, "Critic loss = ", agent.critic_loss)
                    print("Action = ", action)
                    if args.agent == "TD3":
                        print(np.unique(agent.actor.l1.weight.data)[-4:], np.unique(agent.critic.l1_1.weight.data)[-4:])
                    elif args.agent == "GraphActorTD3":
                        print(np.unique(agent.actor.gcn1.weight.data)[-4:], np.unique(agent.critic.l1_1.weight.data)[-4:])
                    elif args.agent == "GCActorTD3v2":
                        print(np.unique(agent.actor.l1_1.weight.data)[-4:], np.unique(agent.critic.l1_1.weight.data)[-4:])
                    else:
                        print(np.unique(agent.actor.gcn.gc3.weight.data)[-4:], np.unique(agent.critic.l1_1.weight.data)[-4:])
        except KeyboardInterrupt:
            # save_np = []
            # for i in range(replay_buffer.size):
            #     save_np.append([replay_buffer.state[i], replay_buffer.action[i], replay_buffer.next_state[i], replay_buffer.reward[i], replay_buffer.not_done[i]])
            # save_np = np.array(save_np)
            # np.save(replaybuffer_mem, save_np)
            print(replay_buffer.size)
def main():

    # Create new threads
    thread = Publisher(pubHipR, pubHipL, pubKneeR, pubKneeL, pubAnkelR, pubAnkelL, rate)

    # Start new Threads
    thread.start()
    listener()

if __name__ == '__main__':
    main()