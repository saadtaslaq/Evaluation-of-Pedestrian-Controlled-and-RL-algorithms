import numpy as np
from .policy import Policy
from crowd_sim.envs.utils.action import ActionXY, ActionRot
import os
from pkg_resources import resource_filename
import PySocialForce.pysocialforce as psf
from crowd_sim.envs import crowd_sim



class PYSF(Policy):
    def __init__(self, safety_space=0):
     
        super().__init__()
        self.name = 'PYSF'
        self.trainable = False
        self.safety_space = safety_space
        self.sim = None


    def set_phase(self, phase):
        self.phase = phase
        return

 
    def reset(self):
        del self.sim
        self.sim = None
    
    # SAAD CHANGES
    # def predict(self, state):
    def predict(self, state_sf, global_map, observation_array, obstacles_array, agent):
        """
        Produce action for agent with circular specification of social force model.
        """
       
        # print("PYSF:--state", state_sf)
        # print("PYSF:--global_map", global_map)
        print("PYSF:--observation_array", observation_array)
        print("PYSF:--obstacles_array", obstacles_array)
        print("PYSF:--agent", agent)
        print("PYSF:--obstacles_array_2 before", obstacles_array)

        # for obstacle in global_map:
        #     print("PYSF:-obstacle-->", obstacle)
        config_dir = resource_filename('crowd_nav', 'config')


        print("PYSF:--obstacles_array_22 after", obstacles_array)
                    
        sim = psf.Simulator(
            state = observation_array,
            groups=None,
            obstacles=obstacles_array,
            # obstacles = obstacles_array_22,
            config_file = os.path.join(config_dir, 'example.toml')
        )
        # update 50 steps
        # action = sim.step(50)
        # print("action PSYF", action)
        # forces = sim.compute_forces()
        # print("force PYSF", forces)
        desired_velocity = sim.step_once()
        print("desired_velocity PYSF", desired_velocity[:, 0:2])
        print("goal for PYSF", desired_velocity[:, 4:6])
       
        # print("next_state PYSF", next_state)
        # n = 50
        desired_velocity =  desired_velocity[:, 0:2]

        print("desired_velocity BEFORE", desired_velocity)

        for i, vel in enumerate(desired_velocity):
            
            vx = vel[0]
            vy = vel[1]
            print("deasiredVel vx PYSF", vx)
            print("deasiredVel vy PYSF", vy)

            # forces_list = sim.forces()
            # print("force list PYSF", forces_list)
            action = ActionXY(vx, vy)
            print("action PYSF----", action)

            return action
