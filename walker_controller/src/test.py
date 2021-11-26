from GCTD3 import GCTD3
from GCActorTD3 import GCActorTD3
from GCActorTD3v2 import GCActorTD3v2
from TD3 import TD3
from GraphActorTD3 import GraphActorTD3
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
parser.add_argument("--batch_size", default=64, type = int)
parser.add_argument("--agent", default="GCTD3")
args = parser.parse_args()

FILE_NAME = args.file_name
EPISODES = 500000
TEST = 5
BUFFER_SIZE = 1000000
START_TRAINING = 10000#args.batch_size*10 #10000

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



env = [(7,2), 6] # [state_dim, action_dim] (8 is number of nodes, 2 is number of feature)
parent_child = ((1,0),(2,0),(1,2),(2,1),(3,1),(4,2),(5,3),(6,4)) # delete (2,1)
Adj_matrix = utils.get_adjacency_matrix(env[0][0], parent_child)
if args.agent == "GCTD3":
    agent = GCTD3(env[0][0], env[0][1], env[1], Adj_matrix)
    FILE_NAME = "GCTD3_Walking_Robot"
    freq = 50 # 18, 0.055s
if args.agent == "GCActorTD3":
    agent = GCActorTD3(env[0][0], env[0][1], env[1], Adj_matrix)
    FILE_NAME = "GCActorTD3_Walking_Robot"
    PATH_LOAD = "GCActorTD3v2_3/GCActorTD3_Walking_Robot_final"
    freq = 50 #32
if args.agent == "GCActorTD3v2":
    agent = GCActorTD3v2(env[0][0], env[0][1], env[1], Adj_matrix)
    FILE_NAME = "GCActorTD3_Walking_Robotv2"
    freq = 50 #32
if args.agent == "GraphActorTD3":
    agent = GraphActorTD3(env[0][0], env[0][1], env[1], Adj_matrix)
    FILE_NAME = "GraphActorTD3_Walking_Robot"
    freq = 50
if args.agent == "TD3":
    agent = TD3(env[0][0]*env[0][1], env[1])
    FILE_NAME = "TD3_Walking_Robot"
    freq = 50
# load and save model
save = False
if args.load_model == True:
    agent.load_model(PATH_LOAD)
    START_TRAINING = 20000
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
        self.robot_state = [[self.vel_y, self.vel_z], [self.hipr_theta, self.hipr_theta_dot], [self.hipl_theta, self.hipl_theta_dot], \
        [self.kneer_theta, self.kneer_theta_dot], [self.kneel_theta, self.kneel_theta_dot], [self.ankelr_theta, self.ankelr_theta_dot], \
        [self.ankell_theta, self.ankell_theta_dot]]

        self.latest_reward = 0.0
        self.best_reward = -100000000000000.0
        self.episode = 0
        self.last_outer_ring_inner_ring_theta = 0.0
        self.last_time = 0.0

        self.fall = 0
        self.done = False
        self.avg_reward = 0.0
        self.count_state = 0
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

    # print "called reset()"



def set_robot_state():
    robot_state.robot_state = [[robot_state.vel_y, robot_state.vel_z], [robot_state.hipr_theta, robot_state.hipr_theta_dot], [robot_state.hipl_theta, robot_state.hipl_theta_dot], \
        [robot_state.kneer_theta, robot_state.kneer_theta_dot], [robot_state.kneel_theta, robot_state.kneel_theta_dot], [robot_state.ankelr_theta, robot_state.ankelr_theta_dot], \
        [robot_state.ankell_theta, robot_state.ankell_theta_dot]]

def take_action_ver2():
    rospy.wait_for_service('/gazebo/unpause_physics')

    try:
        unpause()
    except (rospy.ServiceException) as e:
        print("/gazebo/pause_physics service call failed")

    rate.sleep()
    return None

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

        robot_state.waist_z = data.position[0]
        robot_state.waist_y = data.position[1]
        robot_state.outer_ring_inner_ring_theta = data.position[1]
        robot_state.ankell_theta = data.position[2]
        robot_state.ankelr_theta = data.position[3]
        robot_state.kneel_theta = data.position[4]
        robot_state.kneer_theta = data.position[5]
        robot_state.hipl_theta = data.position[6]
        robot_state.hipr_theta = data.position[7]
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
        take_action_ver2()

def main():

    # Create new threads
    thread = Publisher(pubHipR, pubHipL, pubKneeR, pubKneeL, pubAnkelR, pubAnkelL, rate)

    # Start new Threads
    thread.start()
    listener()

if __name__ == '__main__':
    main()