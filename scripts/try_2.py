import os

# import gymnasium as gym
# import numpy as np 
from pyrep import PyRep
from pyrep.robots.arms.ur5 import UR5
import cv2

# from pyrep.robots.arms.panda import Panda
# from pyrep.robots.end_effectors.panda_gripper import PandaGripper


# pr = PyRep()
# # Launch the application with a scene file that contains a robot
# # pr.launch('./sim/broken_w_all_parts.ttt') 
# pr.launch('./sim/non_pure_non_convex_min_parts.ttt') 
# # pr.launch('task_design.ttt') 
# pr.start()  # Start the simulation



# # 1. Load Environment and Q-table structure
# # env = gym.make('FrozenLake8x8-v0')
# env = pr
# Q = np.zeros([env.observation_space.n,env.action_space.n])
# # env.observation.n, env.action_space.n gives number of states and action in env loaded
# # 2. Parameters of Q-learning
# eta = .628
# gma = .9
# epis = 5000
# rev_list = [] # rewards per episode calculate
# # 3. Q-learning Algorithm
# for i in range(epis):
#     # Reset environment
#     s = env.reset()
#     rAll = 0
#     d = False
#     j = 0
#     #The Q-Table learning algorithm
#     while j < 99:
#         # env.render()
#         j+=1
#         # Choose action from Q table
#         a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
#         #Get new state & reward from environment
#         s1,r,d,_ = env.step(a)
#         #Update Q-Table with new knowledge
#         Q[s,a] = Q[s,a] + eta*(r + gma*np.max(Q[s1,:]) - Q[s,a])
#         rAll += r
#         s = s1
#         if d == True:
#             break
#     rev_list.append(rAll)
#     # env.render()
# print("Reward Sum on all episodes " + str(sum(rev_list)/epis))
# print("Final Values Q-Table")
# print(Q)

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.const import TextureMappingMode, RenderMode
from enum import Enum
import numpy as np
import random
from pyrep.objects.shape import Shape


SCENE_FILE = './sim/non_pure_non_convex_min_parts.ttt'
#join(dirname(abspath(__file__)),
#     'scene_reinforcement_learning_env.ttt')
POS_MIN, POS_MAX = [0.8, -0.2, 1.0], [1.0, 0.2, 1.4]
EPISODES = 5
EPISODE_LENGTH = 200

counter = 0

class TrainingType(Enum):
    BASELINE = 1
    VMP = 2
    VMP_GC = 3


class LqrEnv(gym.Env):
    def __init__(self, headless=True, episode_length=150, training_type=TrainingType.BASELINE):#, size, init_state, state_bound):
        # self.init_state = init_state
        # self.size = size 
        # self.action_space = spaces.Box(low=-state_bound, high=state_bound, shape=(size,))
        # self.observation_space = spaces.Box(low=-state_bound, high=state_bound, shape=(size,))
        # self._seed()
        self.pr = PyRep()
        self.pr.launch(SCENE_FILE, headless)
        self.pr.start()
        
        # -------------------------------------------------
        #Fix Err: qt.qpa.plugin: Could not find the Qt platform plugin "xcb" in "/usr/local/lib/python3.8/dist-packages/cv2/qt/plugins" 
        # os.environ['PATH'] = '$PATH:~/Documents/CSCI_699/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04'
        # os.environ['COPPELIASIM_ROOT'] = '~/Documents/CSCI_699/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04'
        # os.environ['LD_LIBRARY_PATH'] = '$LD_LIBRARY_PATH:$COPPELIASIM_ROOT'
        # os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '$COPPELIASIM_ROOT'
        # os.environ['QT_PLUGIN_PATH'] = '~/Documents/CSCI_699/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04/platforms'
        
        self.episode_length = episode_length
        self.training_type = training_type

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
        # Normalize Action space between [-1,1]
        self.action_space = spaces.Box(low=-1., high=1., shape=(6,), dtype=float)

        # --------------------------------------------------
        # Sensors
        self.vision_sensor = VisionSensor("kinect_rgb")
        self.vision_sensor.set_render_mode(render_mode=RenderMode.OPENGL3)
        self.vision_sensor.set_resolution([64,64])

        # -------------------------------------------------
        self.initial_distance = self.reward_distance_to_goal()
        self.initial_orientation = self.reward_orientation()
        [self.pr.step() for i in range(10)] # Stabilize the simulator

        # self.proximity_sensor = ProximitySensor('ROBOTIQ_85_attachProxSensor')
        # self.force_sensor = ForceSensor("Force_sensor")

    def reset(self, seed=None, options=None):
        
        # # We need the following line to seed self.np_random
        # super().reset(seed=seed)

        # # Choose the agent's location uniformly at random
        # self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # # We will sample the target's location randomly until it does not coincide with the agent's location
        # self._target_location = self._agent_location
        # while np.array_equal(self._target_location, self._agent_location):
        #     self._target_location = self.np_random.integers(
        #         0, self.size, size=len(), dtype=int
        #     )

        # observation = self._get_obs()
        # info = self._get_info()

        # # if self.render_mode == "human":
        # #     self._render_frame()
        # return observation, info

        print("------>",self._get_state())
        self.setup_scene()
        return self._get_state()

    def step(self, action):
        # costs = np.sum(u**2) + np.sum(self.state**2)
        # self.state = np.clip(self.state + u, self.observation_space.low, self.observation_space.high)
        # costs = self.get_costs()
        
        # self.pr.step()
        # return self._get_obs(), -costs, False, {}
    
        done = False
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
        self.agent.set_joint_target_velocities(denorm_action) # Try this
        self.pr.step()  # Step the physics simulation

        #------------------------------------------------
        # Reward calculations

        success_reward, success = self.reward_success()
        distance_reward = (prev_distance_to_goal - self.reward_distance_to_goal())/self.initial_distance # Relative distance reward
        orientation_reward = (previous_orientation_reward - self.reward_orientation())/self.initial_orientation # Relative orientation reward
        reward = (distance_reward * 10) + success_reward + orientation_reward

        #------------------------------------------------
        if self.step_counter % self.episode_length == 0:
            done = True
            info = {"Cause":"Timeout"}
            print('--------Reset: Timeout--------')

        if success:
            done = True
            info = {"Cause":"Success"}
            print('--------Reset: Success is true--------')

        if self.dining_table.check_collision() or self.door.check_collision(obj=self.agent) \
            or self.gripper.check_collision(obj=self.door) or self.gripper.check_collision(obj=self.handle) \
            or abs(self.force_sensor.read()[0][2]) > 100:
            done = True
            info = {"Cause":"Collision"}
            reward += -10
            print(self.gripper.check_collision(obj=self.handle)," ",self.gripper.check_collision(obj=self.door))
            print("----- Reset: Collision -----")

        self.step_counter += 1
        # print(orientation_reward)
        print(self.step_counter," ",distance_reward*10," ",success_reward," ",orientation_reward," ",reward)
        return self._get_state(),reward,done,info


    # def _seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]

    def _get_state(self):
        # Return state containing arm joint angles/velocities & target position
        # return np.concatenate([self.agent.get_joint_positions(),
        #                         self.agent.get_joint_velocities(),
        #                         self.target.get_position()])

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


        elif self.training_type == TrainingType.VMP_GC:
            # VMP current image
            current_image = self.vision_sensor.capture_rgb()
            current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
            # cv2.imwrite("./"+ str(UUID_val)+"_"+str(counter)+".jpg",current_image*255)
            current_representation = self.get_representation(current_image).flatten()

            # Goal image
            goal_image = random.sample(goal_img_list,1)[0]
            # cv2.imwrite("./"+ str(UUID_val)+"_goal_"+str(counter)+".jpg",goal_image)
            goal_representation = self.get_representation(goal_image/255).flatten()

            joint_pos = self.agent.get_joint_positions()
            final_state = np.concatenate((joint_pos , current_representation , goal_representation))

        # return final_state

        return {"agent": self.agent.get_joint_positions(), "target": self.target.get_position()}

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
        self.agent.set_joint_target_velocities([0,0,0,0,0,0])
        self.agent.set_joint_target_positions([0,0,0,0,0,0])

        return reward

    def reward_orientation(self):
        target_orientation = self.target.get_orientation()
        agent_orientation = self.agent_ee_tip.get_orientation(relative_to=self.target)

        orientation_value = (np.cos((agent_orientation[2])))

        return orientation_value


    def reward_success(self):
        DISTANCE = 0.03
        success_reward = -1 # default reward per timestep
        success = False

        if self.proximity_sensor.read() < DISTANCE and \
                self.proximity_sensor.read() != -1 and \
                self.proximity_sensor.is_detected(self.handle):
            success_reward = +10.0
            success = True

        return success_reward, success
    
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
    # from stable_baselines3.common.policies import MlpPolicy
    # from stable_baselines3 import PPO2, A2C

    # env = LqrEnv()
    # model = A2C(MlpPolicy, env, verbose=1)
    # model.learn(total_timesteps=25000)
    # model.save("a2c_robotic_arm")

    # import gymnasium as gym

    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    # Parallel environments
    vec_env = make_vec_env(LqrEnv, n_envs=1)

    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=25000)
    model.save("ppo_cartpole")

    # del model # remove to demonstrate saving and loading

    model = PPO.load("ppo_cartpole")

    obs = vec_env.reset()
    for i in range (100):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        # vec_env.render("human")
    vec_env.shutdown()


    # # env = GridworldEnv(5)
    # env = LqrEnv()
    # states = env.observation_space.shape
    # actions = env.action_space.n
    # model = build_model(states, actions)

    # dqn = build_agent(model, actions)
    # dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    # dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)

    # scores = dqn.test(env, nb_episodes=2, visualize=False)
