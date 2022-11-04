# from httplib2 import Self
from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.policy.social_force import SOCIAL_FORCE
import numpy as np
import random
from crowd_nav.policy.policy_factory import policy_factory


class Human(Agent):
    def __init__(self, config, section):
        self.last_state = None
        super().__init__(config, section)
        self.random_policy_changing = config.get('env', 'random_policy_changing')
        self.policy_name = config.get('env', 'human_policy')
        # self.isObstacle = True # whether the human is a static obstacle (part of wall) or a moving agent


    def act(self, ob=None, global_map=None, local_map=None, observation_array = None, obstacles_array = None):
        """
        The state for human is its full state and all other agents' observable states
        :param ob:
        :return:
        """
        if ob is None:
            return self.policy.predict(self)
        state = JointState(self.get_full_state(), ob)
        state_sf = JointState(self.get_full_state_sf(), ob)
        # print("STATES HUMANS", state)

        
        if global_map is not None:
                        ### FOR DEFAULT OR OTHER POLICIES ###
            
            action = self.policy.predict(state, global_map, self)

                        ### FOR DEFAULT OR SOCIAL FORCES POLICIES ###

            # action = self.policy.predict(state, global_map, obstacles_array, self)

                        ## FOR PYSF ##

            # action = self.policy.predict(state_sf, global_map, observation_array, obstacles_array,  self)

            # print("action --", action)
            
        elif local_map is not None:
            action = self.policy.predict(state, local_map, self)
        else:
            action = self.policy.predict(state)

        return action

    def set_policy(self, policy):
        self.policy = policy

    def set_policy(self, policy):
        if self.random_policy_changing=='True':
            new_policy = random.choice(['orca','social_forces', 'linear'])
            self.policy_name = new_policy
            new_policy = policy_factory[new_policy]()
            print("new_policy", new_policy)
            self.policy = new_policy
        else:
            
            self.policy = policy