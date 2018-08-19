import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None,
        init_angle_velocities=None, runtime=5., target_pos=None,
        stop_on_target=False):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        self.stop_on_target = stop_on_target
        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

    def on_target(self):
        if (abs(self.sim.pose[:3] - self.target_pos)).sum() < 1:
            return True
        return False

    def has_crashed(self):
        if self.sim.done:
            if self.sim.time < self.sim.runtime:
                return True
        return False

    def target_distance(self):
        return (abs(self.sim.pose[:3] - self.target_pos)).sum()

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()

        if self.on_target() and self.stop_on_target is True:
            reward += 100000

        if self.has_crashed():
            reward -= 100000
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        pose_all = []
        reward = 0
        for _ in range(self.action_repeat):
        # for _ in [1]:
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
            if self.stop_on_target and self.on_target():
                done = True
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state


LAND = [0., 0., 0.]
HOVER = [0., 0., 10.]
TAKEOFF = [0., 0., 20.]
OFFSET = [10., 10., 20.]
runtime = 5.

init_pose = np.array([0., 0., 10., 0., 0., 0.])  # initial pose
init_velocities = np.array([0., 0., 0.])         # initial velocities
init_angle_velocities = np.array([0., 0., 0.])   # initial angle velocities

def LandTask():
    return Task(init_pose, init_velocities, init_angle_velocities, runtime, target_pos=LAND, stop_on_target=False)

def HoverTask():
    return Task(np.array(HOVER), init_velocities, init_angle_velocities, runtime, target_pos=HOVER, stop_on_target=False)

def TakeoffTask():
    return Task(init_pose, init_velocities, init_angle_velocities, runtime, target_pos=TAKEOFF, stop_on_target=True)

def OffsetTask():
    return Task(init_pose, init_velocities, init_angle_velocities, runtime, target_pos=OFFSET, stop_on_target=False)
