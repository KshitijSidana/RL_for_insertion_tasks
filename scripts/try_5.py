import os
import cv2
import random
import numpy as np
from enum import Enum

# from gym import spaces

import gymnasium as gym
from gymnasium import spaces
# from gymnasium.spaces import box
from gymnasium.utils import seeding

from pyrep import PyRep
from pyrep.objects.shape import Shape, collections
from pyrep.robots.arms.ur5 import UR5
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.const import TextureMappingMode, RenderMode
from pyrep.backend import sim

from stable_baselines3.sac import SAC
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.logger import configure

SCENE_FILE = './sim/non_pure_non_convex_min_parts.ttt'
POS_MIN, POS_MAX = [0.8, -0.2, 1.0], [1.0, 0.2, 1.4]
EPISODES = 2
EPISODE_LENGTH = 200

counter = 0

class TrainingType(Enum):
    BASELINE = 1
    VMP = 2
    VMP_GC = 3


class LqrEnv(gym.Env):
    def __init__(self, headless=False, episode_length=150, training_type=TrainingType.BASELINE):#, size, init_state, state_bound):
        # self.init_state = init_state
        # self.size = size 
        # self.action_space = spaces.Box(low=-state_bound, high=state_bound, shape=(size,))
        # self.observation_space = spaces.Box(low=-state_bound, high=state_bound, shape=(size,))
        # self._seed()
        self.pr = PyRep()
        self.pr.launch(SCENE_FILE, headless)
        self.pr.start()
        
        self.episode_length = episode_length
        self.training_type = training_type
        self.step_counter = 0
        
        # -------------------------------------------------
        # Initialize VecNormalize parameters manually
        self.norm_obs = True
        self.norm_reward = False
        self.clip_obs = 255.0  # Adjust clip_obs as needed
        
        # -------------------------------------------------
        # Creating the Robot and Other objects
        self.agent = UR5()
        self.agent.set_control_loop_enabled(False)
        self.agent.set_motor_locked_at_zero_velocity(True)
        self.target = Shape('nacs_1kv_port')
        self.agent_ee_tip = self.agent.get_tip()
        self.initial_joint_positions = self.agent.get_joint_positions()

        # -------------------------------------------------
        # Representation Generation
        # #  TODO !!
        # self.ae_model = LangeConvAutoencoder.load_from_checkpoint(AE_MODEL_PATH)
        # self.ae_model.eval()

        # -------------------------------------------------
        # Setting State space for robot
        self.observation_space = None
        if self.training_type == TrainingType.BASELINE:
            self.observation_space = spaces.Box(low=np.asarray([val[0] for val in self.agent.get_joint_intervals()[1]]),
                                        high=np.asarray([val[1] for val in self.agent.get_joint_intervals()[1]]), dtype=float)
        elif self.training_type == TrainingType.VMP:
            self.observation_space = spaces.Box(low=np.asarray([val[0] for val in self.agent.get_joint_intervals()[1]] + [0. for i in range(14)]),
                                        high=np.asarray([val[1] for val in self.agent.get_joint_intervals()[1]] + [1. for i in range(14)]), dtype=float)
        elif self.training_type == TrainingType.VMP_GC:
            self.observation_space = spaces.Box(low=np.asarray([val[0] for val in self.agent.get_joint_intervals()[1]] + [0. for i in range(28)]),
                                        high=np.asarray([val[1] for val in self.agent.get_joint_intervals()[1]] + [1. for i in range(28)]), dtype=float)

        # -------------------------------------------------
        # # Normalize Action space between [-1,1]
        # self.action_space = box.Box(low=-1., high=1., shape=(6,), dtype=float)
        # Normalize Action space between [-1,1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=float)
        # self.action_space = self.agent.action_space()
        # --------------------------------------------------
        # Sensors
        self.vision_sensor = VisionSensor("kinect_rgb")
        self.vision_sensor.set_render_mode(render_mode=RenderMode.OPENGL3)
        self.vision_sensor.set_resolution([64,64])

        # -------------------------------------------------
        self.initial_distance = self.reward_distance_to_goal()
        self.initial_orientation = self.reward_orientation()
        [self.pr.step() for i in range(10)] # Stabilize the simulator

    def setup_scene(self):
        # Reset the robot's joint positions

        self.agent.set_joint_positions(self.initial_joint_positions)
        # self.target.set_position()

        # You may need additional code here to reset or initialize other objects in the scene
        # For example, if there are target positions, you might reset them.

        # Stabilize the simulator
        [self.pr.step() for i in range(10)]

        # Return the initial state after setup
        return self._get_state()
    
    def reset(self, seed=None, options=None):
        state = self._get_state()
        print("------>",state)
        temp =  self.step_counter
        self.step_counter = 0
        self.setup_scene()
        
        if self.norm_obs:
            state = self.normalize_observation(state)
        
        return state, temp

    def step(self, action):
        done = False
        truncated = False
        info = {}
        prev_distance_to_goal = self.reward_distance_to_goal()
        previous_orientation_reward = self.reward_orientation()

        #-----------------------------------------------
        # Denorm  action values
        lower_limits = [-val for val in self.agent.get_joint_upper_velocity_limits()]
        upper_limits = [val for val in self.agent.get_joint_upper_velocity_limits()]

        denorm_action = []

        for num,low,high in zip(action,lower_limits,upper_limits):
            new = np.interp(num,[-1,1],[low,high])
            denorm_action.append(new)

        # denorm_action[:6] = [0,0,0,0,0,0]
        # self.agent.set_joint_target_velocities(denorm_action) # Try this
        # self.agent.set_joint_positions([0.5,0.5,0.5,0.5,0.5,0.5]) # Try this
        self.agent.set_joint_positions(denorm_action) # Try this
        self.pr.step()  # Step the physics simulation

        #------------------------------------------------
        # Reward calculations

        distance, success_reward, success = self.reward_success()
        distance_reward = (distance - prev_distance_to_goal)/self.initial_distance # Relative distance reward
        # orientation_reward = (previous_orientation_reward - self.reward_orientation())/self.initial_orientation # Relative orientation reward
        reward = (distance_reward * 10) + success_reward #+ orientation_reward

        #------------------------------------------------
        if self.step_counter % self.episode_length == 0:
            done = True
            truncated = True
            info = {"Cause":"Timeout"}
            print('--------Reset: Timeout--------')

        if success:
            done = True
            info = {"Cause":"Success"}
            print('--------Reset: Success is true--------')

        # collision_plug_port = self.pr. simCheckCollision(self.pr.get_collection_handle_by_name('Plug'))
        # Shape(self.pr.get_collection_handle_by_name('Port')).check_collision(Shape(self.pr.get_collection_handle_by_name('Port')))

        collision_agent_wall, collision_agent_port, collision_plug, collision_port, collision_floor = False, False, False, False, False  

        collision_agent_wall = self.agent.check_arm_collision(Shape('Wall')) 
        
        for no in [0, 1, 2, 3, 6, 7, 8, 21, 22, 23, 24, 25, 27]:
            collision_port = self.agent.check_arm_collision(Shape('Cuboid'+str(no)))
            if collision_port:
                print(f"Cuboid{no}") 
            collision_agent_port = collision_agent_port or collision_port
        # collision_port = sim.simCheckCollision(self.pr.get_collection_handle_by_name('Port'), sim.sim_handle_all)
        # collision_plug = sim.simCheckCollision(self.pr.get_collection_handle_by_name('Plug'), sim.sim_handle_all)

        if collision_agent_wall or collision_agent_port or collision_plug or collision_port:

        # if self.dining_table.check_collision() or self.door.check_collision(obj=self.agent) \
            # or self.gripper.check_collision(obj=self.door) or self.gripper.check_collision(obj=self.handle) \
            # or abs(self.force_sensor.read()[0][2]) > 100:
            done = True
            info = {"Cause":"Collision"}
            reward += -100
            print(f"collision_agent_port {collision_agent_port}, collision_agent_wall {collision_agent_wall}, collision_plug {collision_plug}, collision_port {collision_port}, collision_floor {collision_floor}")
            print("----- Reset: Collision -----")

        self.step_counter += 1
        # print(orientation_reward)
        print(f"tep_counter {self.step_counter}, distance*10 {distance_reward*10}, success_reward {success_reward}, orientation_reward n/a  reward {reward}")
        # return self._get_state(),reward,done,info
            
        
        if self.norm_obs:
            next_state = self._get_state()
            next_state = self.normalize_observation(next_state)
        else:
            next_state = self._get_state()
        
        
        return self._get_state(),reward,done,truncated,info
    
    def normalize_observation(self, obs):
        # Normalize the values to a specific range
        obs_low = np.asarray([val[0] for val in self.agent.get_joint_intervals()[1]])
        obs_high = np.asarray([val[1] for val in self.agent.get_joint_intervals()[1]])
        
        # normalized_obs = {"agent": (obs["agent"] - obs_low) / (obs_high - obs_low), "target": self.target.get_position()}  
        normalized_obs = (obs - obs_low) / (obs_high - obs_low)  
        
        return normalized_obs


    # def _seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]

    def _get_state(self):
        # Return state containing arm joint angles/velocities & target position
        global counter
        counter += 1

        final_state = None
        if self.training_type == TrainingType.BASELINE:
            joint_pos = self.agent.get_joint_positions()
            final_state = np.concatenate([joint_pos])

        elif self.training_type == TrainingType.VMP:
            # VMP current image
            current_image = self.vision_sensor.capture_rgb()
            current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
            current_representation = self.get_representation(current_image).flatten()

            joint_pos = self.agent.get_joint_positions()
            final_state = np.concatenate((joint_pos , current_representation))


        # elif self.training_type == TrainingType.VMP_GC:
        #     # VMP current image
        #     current_image = self.vision_sensor.capture_rgb()
        #     current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
        #     cv2.imwrite("./"+ str(UUID_val)+"_"+str(counter)+".jpg",current_image*255)
        #     current_representation = self.get_representation(current_image).flatten()

        #     # Goal image
        #     goal_image = random.sample(goal_img_list,1)[0]
        #     # cv2.imwrite("./"+ str(UUID_val)+"_goal_"+str(counter)+".jpg",goal_image)
        #     goal_representation = self.get_representation(goal_image/255).flatten()

        #     joint_pos = self.agent.get_joint_positions()
        #     final_state = np.concatenate((joint_pos , current_representation , goal_representation))

        return final_state

        # return {"agent": self.agent.get_joint_positions(), "target": self.target.get_position()}
        # return self.agent.get_joint_positions()

    def _get_info(self):
        return {"distance": np.linalg.norm(self.agent.get_joint_positions() - self._target_location, ord=1)}
    

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}
    
    
    # ---------------------------REWARDS-----------------------------
    def reward_distance_to_goal(self):
        # Reward is negative distance to target
        agent_position = self.agent_ee_tip.get_position()
        target_position = self.target.get_position()

        reward = np.linalg.norm(agent_position - target_position)

        # Reset environment with initial conditions
        self.agent.set_joint_target_velocities([0,10,10,10,0,0])
        # self.agent.set_joint_target_positions([0,0,0,0,0,0])

        return reward

    def reward_orientation(self):
        target_orientation = self.target.get_orientation()
        agent_orientation = self.agent_ee_tip.get_orientation(relative_to=self.target)

        orientation_value = (np.cos((agent_orientation[2])))

        return orientation_value


    # def reward_success(self):
    #     DISTANCE = 0.03
    #     success_reward = -1 # default reward per timestep
    #     success = False

    #     if self.proximity_sensor.read() < DISTANCE and \
    #             self.proximity_sensor.read() != -1 and \
    #             self.proximity_sensor.is_detected(self.handle):
    #         success_reward = +10.0
    #         success = True

    #     return success_reward, success
    
    def reward_success(self):
        DISTANCE = 0.03
        success_reward = -1 # default reward per timestep
        success = False
        
        # Distance
        ###> Top
        d_top = 0.0
        d_top += Shape('Cuboid1').check_distance(Shape('Cuboid28'))
        d_top += Shape('Cuboid3').check_distance(Shape('Cuboid26'))
        
        ###> Right
        d_right = 0.0
        d_right += Shape('Cuboid2').check_distance(Shape('Cuboid17'))
        
        ###> Left
        d_left = 0.0
        d_left += Shape('Cuboid0' ).check_distance(Shape('Cuboid9'))
        
        ###> Bottom 
        d_bottom = 0.0
        d_bottom += Shape('Cuboid6').check_distance(Shape('Cuboid31'))
        d_bottom += Shape('Cuboid7').check_distance(Shape('Cuboid30'))
        
        
        d_total = sum([d_top, d_right, d_left, d_bottom])
        distance_reward = -1*d_total + DISTANCE*4

        
        if d_total < DISTANCE*4 :
            success_reward = +10.0
            success = True

        return distance_reward, success_reward, success


    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()

    def set_agent_pos(self):
        pass

    def set_target_pos(self):
        pass


class Agent(object):

    def act(self, state):
        del state
        return list(np.random.uniform(-1.0, 1.0, size=(7,)))

    def learn(self, replay_buffer):
        del replay_buffer
        pass


if __name__ == "__main__":

    NAME = "PPO_MlpPolicy_000"

    env = LqrEnv(episode_length=300)
    # Assuming 'env' is your environment
    env = Monitor(env)  # Wrap with Monitor if needed
    env = DummyVecEnv([lambda: env])  # Wrap with DummyVecEnv

    # Use VecNormalize for normalizing observations
    # env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=255.0)
    tmp_path = "tmp/sb3_log/"
    # set up logger
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model = PPO("MlpPolicy", env, verbose=1)
    # model = SAC(MlpPolicy, env, verbose=1)
    # model.load("TD3_000")
    model.set_logger(new_logger)
    
    model.learn(total_timesteps=25000, progress_bar = True)
    model.save(NAME)