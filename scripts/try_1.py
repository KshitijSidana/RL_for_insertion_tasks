from pyrep import PyRep
from pyrep.robots.arms.ur5 import UR5
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
from pyrep import const
pr = PyRep()
# Launch the application with a scene file that contains a robot
# pr.launch('./sim/broken_w_all_parts.ttt') 
pr.launch('./sim/non_pure_non_convex_min_parts.ttt') 
# pr.launch('task_design.ttt') 
pr.start()  # Start the simulation

arm_5 = UR5()  # Get the panda from the scene
# arm = Panda()  # Get the panda from the scene
# gripper = PandaGripper()  # Get the panda gripper from the scene

velocities = [1, 1, 1, 1, 1, 1]
velocities = [0 for _ in range(6)]
arm_5.set_joint_target_velocities(velocities)
arm_5.set_joint_positions([10.1, 10.10, 10.1, 10.4, 10.5, 10.1])
arm_5.set_joint_positions([0,0,0,0,0,0])

pr.step()  # Step physics simulation

arm_5.set_joint_mode(const.JointMode.FORCE)

done = False
# # Open the gripper halfway at a velocity of 0.04.
for i in range(50):
    arm_5.set_joint_target_positions([0,0,i/2,0,1,0])
    # arm_5.set_joint_target_velocities([i for _ in range(6)])
    # done = gripper.actuate(0.5, velocity=0.04)
    pr.step()
    
    if done:
        break
    
pr.stop()  # Stop the simulation
pr.shutdown()  # Close the application0