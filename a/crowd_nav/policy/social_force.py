import numpy as np
from a.crowd_sim.envs.policy.policy import Policy
from a.crowd_sim.envs.utils.action import ActionXY


class SOCIAL_FORCE(Policy):
    def __init__(self, safety_space=0):
        # super().__init__(config)
     
        super().__init__()
        self.name = 'social_forces'
        self.trainable = False
        self.safety_space = safety_space
        self.neighbor_dist = 10
        self.max_neighbors = 10
        self.time_horizon = 5
        self.time_horizon_obst = 5
        self.sim = None
        self.FOV_min_angle = -np.pi % (2 * np.pi)
        self.FOV_max_angle = np.pi % (2 * np.pi)



    def set_phase(self, phase):
        self.phase = phase
        return

    def configure(self, config):
        self.FOV_min_angle = config.getfloat(
            'map', 'angle_min') * np.pi % (2 * np.pi)
        self.FOV_max_angle = config.getfloat(
            'map', 'angle_max') * np.pi % (2 * np.pi)
        self.safety_space = config.getfloat('reward', 'discomfort_dist')
        self.config.env.time_step = config.getint('env', 'time_step')

    def reset(self):
        del self.sim
        self.sim = None
    
    # SAAD CHANGES
    # def predict(self, state):
    def predict(self, state, global_map, agent):#put obstacles as an arg
        """
        Produce action for agent with circular specification of social force model.
        """
        self_state = state.self_state

        # Pull force to goal
        delta_x = state.self_state.gx - state.self_state.px
        delta_y = state.self_state.gy - state.self_state.py
        dist_to_goal = np.sqrt(delta_x**2 + delta_y**2)
        desired_vx = (delta_x / dist_to_goal) * state.self_state.v_pref
        desired_vy = (delta_y / dist_to_goal) * state.self_state.v_pref
        KI = 1 #self.config.sf.KI # Inverse of relocation time K_i
        curr_delta_vx = KI * (desired_vx - state.self_state.vx)
        curr_delta_vy = KI * (desired_vy - state.self_state.vy)
        
        # Push force(s) from other agents
        A = 2 #self.config.sf.A # Other observations' interaction strength: 1.5
        B = 1 #self.config.sf.B # Other observations' interaction range: 1.0
        interaction_vx = 0
        interaction_vy = 0
        for other_human_state in state.human_states:
            print("state.human_states", state.human_states)
            delta_x = state.self_state.px - other_human_state.px
            delta_y = state.self_state.py - other_human_state.py
            dist_to_human = np.sqrt(delta_x**2 + delta_y**2)
            interaction_vx += A * np.exp((state.self_state.radius + other_human_state.radius - dist_to_human) / B) * (delta_x / dist_to_human)
            interaction_vy += A * np.exp((state.self_state.radius + other_human_state.radius - dist_to_human) / B) * (delta_y / dist_to_human)

        #Push force from nearest obstacle(We Need Obstacle_Array)
        #Find the nearest location of obstacle
        #Base on the distance between Human and Obst, Obst direction to the Huamn, get the force
        
    


        #CHANGE BELOW: ADD Push force from Obst into SUM

        # Sum of push & pull forces
        total_delta_vx = (curr_delta_vx + interaction_vx) * 0.2#self.config.env.time_step
        total_delta_vy = (curr_delta_vy + interaction_vy) * 0.2#self.config.env.time_step

        # clip the speed so that sqrt(vx^2 + vy^2) <= v_pref
        new_vx = state.self_state.vx + total_delta_vx
        # print('new_vx', new_vx)
        new_vy = state.self_state.vy + total_delta_vy
        # print('new_vy', new_vy)

        act_norm = np.linalg.norm([new_vx, new_vy])
        # print("act_norm", act_norm)
        # print("state.self_state.v_pre", state.self_state.v_pref)

        if act_norm > state.self_state.v_pref:
            # print("ActionXY(new_vx / act_norm * state.self_state.v_pref, new_vy / act_norm * state.self_state.v_pref",ActionXY(new_vx / act_norm * state.self_state.v_pref, new_vy / act_norm * state.self_state.v_pref))
            action = ActionXY(new_vx / act_norm * state.self_state.v_pref, new_vy / act_norm * state.self_state.v_pref)
    

        else:
            # print("ActionXY(new_vx, new_vy)",ActionXY(new_vx, new_vy))

            action = ActionXY(new_vx, new_vy)

        return action

class Linear(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'linear'
        self.trainable = False
        self.kinematics = 'holonomic'
        self.multiagent_training = True

    def set_phase(self, phase):
        self.phase = phase
        return

    def reset(self):
        del self.sim
        self.sim = None

    def configure(self, config):
        assert True

    def predict(self, state, global_map, agent):
        self_state = state.self_state
        theta = np.arctan2(self_state.gy-self_state.py, self_state.gx-self_state.px)
        # pos = self.compute_position()
        # for i, p in enumerate(pos):
        #     for obstacle in global_map:        
        #         diff = p - obstacle;

        vx = np.cos(theta) * self_state.v_pref
        vy = np.sin(theta) * self_state.v_pref
        action = ActionXY(vx, vy)

        return action
