import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None,
        init_angle_velocities=None, runtime=5., target_pos=None,
        stop_on_target=False, rotation_penalty=False, movement_penalty=False, target_penalty=True, crash_penalty=False,
        feature_atom="pose", action_repeat=3):
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

        self.feature_atom = feature_atom
        self.action_repeat = action_repeat

        atom_size = 0
        if feature_atom == "pose":
            atom_size = 6
        elif feature_atom == "velocity":
            atom_size = 3 + 3
        elif  feature_atom == "pose+velocity":
            atom_size = 6 + 3 + 3
        else:
            assert False, "Unkown feature_atom of {}".format(feature_atom)

        self.state_size = self.action_repeat * atom_size
        self.action_low = 1
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

        self.target_penalty = target_penalty
        self.crash_penalty = crash_penalty
        self.rotation_penalty = rotation_penalty
        self.movement_penalty = movement_penalty
        self.stop_on_target = stop_on_target

    def __str__(self):
        s = ""
        s += "{}: action_repeat:{} feature_atom:{} \n".format(type(self), self.action_repeat, self.feature_atom)
        s += "target_penalty: {} movement_penalty: {} rotation_penalty: {} crash_penalty: {}\n".format(self.target_penalty, self.movement_penalty, self.rotation_penalty, self.crash_penalty)
        return s

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
        return np.sqrt(np.power(self.sim.pose[:3] - self.target_pos, np.array([2, 2, 2])).sum())

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 0.
        if self.target_penalty:
            reward += 100./(self.target_distance()+1)

        if self.on_target() and self.stop_on_target is True:
            reward += 100000

        if self.rotation_penalty:
            reward += 10./(.25*np.abs(self.sim.angular_v).sum()+1)

        if self.movement_penalty:
            reward += 10./(.25*np.abs(self.sim.v).sum()+1)

        if self.crash_penalty and self.has_crashed():
             reward -= 1000000

        return reward

    def feature(self):
        if self.feature_atom == "pose":
            return np.concatenate([self.sim.pose])
        elif self.feature_atom == "velocity":
            return np.concatenate([self.sim.v, self.sim.angular_v])
        elif self.feature_atom == "pose+velocity":
            return np.concatenate([self.sim.pose, self.sim.v, self.sim.angular_v])
        else:
            assert False, "self.feature_atom of {} wasn't handled".format(self.feature_atom)

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        pose_all = []
        reward = 0
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.feature())

            if self.stop_on_target and self.on_target():
                done = True
        next_state = np.concatenate(pose_all)
        assert len(next_state) == self.state_size, "{} == {}, {}, {}".format(len(next_state), self.state_size, len(self.feature()), self.action_repeat)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        feature = self.feature()
        state = np.tile(feature, self.action_repeat)
        assert len(state) == self.state_size
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
    return Task(init_pose, init_velocities, init_angle_velocities, runtime, target_pos=LAND, stop_on_target=False, rotation_penalty=True)

def HoverTask(*args, **kwargs):
    kwargs['target_pos'] = HOVER
    if 'feature_atom' not in kwargs:
        kwargs['feature_atom'] = 'velocity'
    if 'target_penalty' not in kwargs:
        kwargs['target_penalty'] = False
    if 'rotation_penalty' not in kwargs:
        kwargs['rotation_penalty'] = True
    if 'movement_penalty' not in kwargs:
        kwargs['movement_penalty'] = True
    if 'stop_on_target' not in kwargs:
        kwargs['stop_on_target'] = False
    if 'crash_penalty' not in kwargs:
        kwargs['crash_penalty'] = False

    return Task(np.array(HOVER + [0., 0., 0.]), init_velocities, init_angle_velocities, runtime, **kwargs)

def TakeoffTask():
    return Task(init_pose, init_velocities, init_angle_velocities, runtime, target_pos=TAKEOFF, stop_on_target=True, rotation_penalty=True)

def OffsetTask():
    return Task(init_pose, init_velocities, init_angle_velocities, runtime, target_pos=OFFSET, stop_on_target=True, rotation_penalty=False)
