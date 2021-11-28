from net.TD3.TD3 import TD3
from net.DDPG.DDPG import DDPG
from net.SAC.SAC import SAC
import gc
gc.enable()


import rospy
import random
import time
import math
import csv
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelConfiguration

from sensor_msgs.msg import JointState
from gazebo_msgs.msg import ContactsState
from std_msgs.msg import Float64

import threading
from scipy.interpolate import interp1d
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--load_model", action="store_true") 
parser.add_argument("--dir_weights", default="weights")
parser.add_argument("--file_name", default="TD3_Walking_Robot")
parser.add_argument("--agent", default="TD3")
args = parser.parse_args()

FILE_NAME = args.file_name
EPISODES = 500000
TEST = 50
NUMBER_HISTORY=5

reward_file = f"net/{args.agent}/reward_file_test.csv"
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

fall = 0


rospy.init_node('walker_control_script')
rate = rospy.Rate(50)

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
        self.ankl_pos = 0.0
        self.ankr_pos = 0.0
        self.robot_state = [self.vel_y, self.vel_z, self.hipr_theta, self.hipr_theta_dot, self.hipl_theta, self.hipl_theta_dot, \
            self.kneer_theta, self.kneer_theta_dot, self.kneel_theta, self.kneel_theta_dot, self.ankelr_theta, self.ankelr_theta_dot, \
            self.ankell_theta, self.ankell_theta_dot, self.footr_contact, self.footl_contact]
        self.robot_state2 = [[self.waist_z, self.vel_z], [self.hipr_theta, self.hipr_theta_dot], [self.hipl_theta, self.hipl_theta_dot], \
            [self.kneer_theta, self.kneer_theta_dot], [self.kneel_theta, self.kneel_theta_dot], [self.ankelr_theta, self.ankelr_theta_dot], \
            [self.ankell_theta, self.ankell_theta_dot], [self.footr_contact, self.footl_contact]]

        self.latest_reward = 0.0
        self.best_reward = -100000000000000.0
        self.episode = 0
        self.last_outer_ring_inner_ring_theta = 0.0
        self.last_time = 0.0

        self.fall = 0
        self.done = False
        self.count_of_1 = 0
        self.avg_reward = 0.0
        self.count_state = 0
        self.count_contactL = 0
        self.count_contactR = 0
        self.count_hipL = 0
        self.count_hipR = 0
        self.signhipR =1
        self.signhipL =1
        self.last_y = self.waist_y


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
    set_robot_state2()

    # print "called reset()"



def set_robot_state():
    robot_state.robot_state = [robot_state.vel_y, robot_state.vel_z, robot_state.hipr_theta, robot_state.hipr_theta_dot, robot_state.hipl_theta, robot_state.hipl_theta_dot, \
        robot_state.kneer_theta, robot_state.kneer_theta_dot, robot_state.kneel_theta, robot_state.kneel_theta_dot, robot_state.ankelr_theta, robot_state.ankelr_theta_dot, \
        robot_state.ankell_theta, robot_state.ankell_theta_dot, robot_state.footr_contact, robot_state.footl_contact]
def set_robot_state2():
    robot_state.robot_state2 = [[robot_state.waist_z, robot_state.vel_z], [robot_state.hipr_theta, robot_state.hipr_theta_dot], [robot_state.hipl_theta, robot_state.hipl_theta_dot], \
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
    delta = y_set[0] - y_set[-1]
    

    reward = -0.1
    reward += min(robot_state.vel_y, 0.9) + 0.2*delta
    

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
        reward -= 0.2
    if robot_state.count_hipL>50 or robot_state.count_hipR > 50:
        robot_state.done = True
        robot_state.fall = 1
        robot_state.count_hipR, robot_state.count_hipL =0, 0
        reward -= 0.5
    
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
    return reward, robot_state.done



def callbackJointStates(data):
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

        robot_state.waist_z = data.position[0]
        robot_state.waist_y = data.position[1]
        robot_state.outer_ring_inner_ring_theta = data.position[1]
        robot_state.ankell_theta = data.velocity[2]
        robot_state.ankelr_theta = data.velocity[3]
        robot_state.kneel_theta = data.position[4]
        robot_state.kneer_theta = data.position[5]
        robot_state.hipl_theta = data.position[6]
        robot_state.hipr_theta = data.position[7]

        robot_state.ankl_pos = data.position[2]
        robot_state.ankr_pos = data.position[3]
    else:
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

        robot_state.ankl_pos = 0
        robot_state.ankr_pos = 0

    set_robot_state()
    set_robot_state2()
    # rate.sleep()


def callbackSub(data):
    set_robot_state()
    set_robot_state2()

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

        # writing rewards in the csv file
        file = open(reward_file, 'wt')
        writer = csv.writer(file)
        writer.writerow(['avg_reward'])

        env = [16, 6] # [state_dim, action_dim]
        env_gcn = [(8,2), 6]
        if args.agent == "TD3":
            agent = TD3(env[0], env[1])
            FILE_NAME = "TD3_Walking_Robot"
        if args.agent == "DDPG":
            agent = DDPG(env[0], env[1])
            FILE_NAME = "DDPG_Walking_Robot"
        if args.agent == "SAC":
            agent = SAC(env[0], env[1])
            FILE_NAME = "SAC_Walking_Robot"
        

        # load and save model
        if args.load_model == True:
            agent.load_model(FILE_NAME, args.agent)
        robot_state.best_reward =200 # manual 

        # Testing:
        traj_file = open(trajectory_file, 'wt')
        traj_writer = csv.writer(traj_file, delimiter='\t')
        traj_writer.writerow(['z', 'y', 'vel_z', 'vel_y', 'hr', 'hl', 'hr_dot', 'hl_dot', 'kr', 'kl', 'kr_dot', 'kl_dot', 'ar', 'al', 'ar_dot', 'al_dot', 'fr', 'fl'])

        print("testing")
        nb_test = 0

        while (nb_test < TEST):
            number_step = 0
            total_reward = 0
            reset()
            state = robot_state.robot_state
            state_tmp = robot_state.robot_state2
            observes = np.concatenate([state_tmp]*NUMBER_HISTORY, axis=1)
            y_set = np.zeros(NUMBER_HISTORY)
            for steps in range(500):    
                number_step += 1
                action = agent.predict_action(np.array(state)) # direct action for test
                reward, done = take_action(action, observes, y_set)
                state = robot_state.robot_state
                state_tmp = robot_state.robot_state2
                observes = np.append(state_tmp, observes[:,:env_gcn[0][1]*(NUMBER_HISTORY-1)], axis=1)
         
                traj_writer.writerow([robot_state.waist_z, robot_state.waist_y, robot_state.vel_z, robot_state.vel_y, robot_state.hipr_theta, robot_state.hipl_theta, robot_state.hipr_theta_dot, robot_state.hipl_theta_dot, \
                                        robot_state.kneer_theta, robot_state.kneel_theta, robot_state.kneer_theta_dot, robot_state.kneel_theta_dot, robot_state.ankr_pos, robot_state.ankl_pos, robot_state.ankelr_theta_dot, \
                                        robot_state.ankell_theta_dot, robot_state.footr_contact, robot_state.footl_contact])
                traj_file.flush()
                y_set = np.append(robot_state.waist_y, y_set[:NUMBER_HISTORY-1])
                total_reward += reward             
                if done:
                    robot_state.done = False
                    break
            ave_reward = total_reward
            robot_state.latest_reward = ave_reward
            if ave_reward > robot_state.best_reward:
                robot_state.best_reward = ave_reward
            robot_state.avg_reward = ave_reward
            if ave_reward > 150:
                nb_test +=1
            else:
                continue
            writer.writerow([ave_reward])
            file.flush()

            print("episode: ",nb_test, "number step: ", number_step,"Evaluation Average Reward: ",ave_reward)
            print("best_reward: ", robot_state.best_reward)
        break

def main():

    # Create new threads
    thread = Publisher(pubHipR, pubHipL, pubKneeR, pubKneeL, pubAnkelR, pubAnkelL, rate)

    # Start new Threads
    thread.start()
    listener()

if __name__ == '__main__':
    main()