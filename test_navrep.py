import os
import numpy as np
from tqdm import tqdm
from stable_baselines import PPO2
from navrep.envs.navreptrainencodedenv import NavRepTrainEncodedEnv

from crowd_sim.envs.utils.info import Timeout, ReachGoal, Danger, Collision, CollisionOtherAgent, Deadlock, Collision_Pedestrian, ReachGoal_Ped
from navrep.tools.commonargs import parse_common_args
from crowd_sim.envs.crowd_sim import CrowdSim  # reference to env code  # noqa

import gym
import logging
from crowd_sim.envs.utils.robot import Robot
from pkg_resources import resource_filename
import configparser
import time

class NavRepCPolicy(object):
   """ wrapper for gym policies """
   def __init__(self, model=None):
       if model is not None:
           self.model = model
       else:
           self.model_path = os.path.expanduser(
               "~/navrep/models/gym/navreptrainencodedenv_2020_09_20__09_56_12_PPO_VAE1DLSTM_V_ONLY_V32M64_ckpt")
           self.model = PPO2.load(self.model_path)
           print("Model '{}' loaded".format(self.model_path))
           # print("AAAAAAAAAAAAAAAA", self.model.observation_space.shape)
 
   def act(self, obs):
       # print("BBBBBBBBBBBBBBBBB", self.model.observation_space.shape)
       action, _states = self.model.predict(obs, deterministic=True)
       return action
 
 
def run_test_episodes(env, policy, render=False, print_failure=True, num_episodes=20):
   success_times = []
   success_ped = 0
   success_ped_times = []
   collision_times = []
   collision_other_agent_times = []
   pedestrian_collision = []
   timeout_times = []
   deadlock_times = []
   computational_time = 0
   success = 0
   collision = 0
   collision_other_agent = 0
   timeout = 0
   deadlock = 0
   too_close = 0
   min_dist = []
   cumulative_rewards = []
   collision_cases = []
   collision_other_agent_cases = []
   timeout_cases = []
   deadlock_cases = []
   progress_bar = tqdm(range(num_episodes), total=num_episodes)
   gamma = 0.9
 
 
#    avg_speed = []
#    speed_violation = []
#    social_violation_cnt = []
#    personal_violation_cnt = []
#    jerk_cost = []
#    aggregated_time = []
#    side_preference = []
#    discomfort = 0
   collision_pedestrian = 0
   for i in progress_bar:
       
       start_time = time.time()

       progress_bar.set_description("Case {}".format(i))
    #    print("progress_bar = ", progress_bar)
    #    with tqdm(total=num_episodes) as t:
    #        ...
    #        t.update()
    #        print("First = ", t.format_interval(t.format_dict['elapsed']))
    #        print("Second = ", str(t).split()[3])

       ob = env.reset()
       done = False
       env_time = 0
       rewards = []
 
 
       episodic_info = {
        #    'discomfort' : 0,
           'min_dist' : [],
           'end_event' : None,
           'deadlock' : 0,
        #    'avg_speed' : 0,
        #    'speed_violation' : 0,
        #    'social_violation_cnt' : 0,
        #    'personal_violation_cnt' : 0,
        #    'jerk_cost' : 0,
        #    'aggregated_time' : 0,
           'side_preference' : None
       }
       states = 0
       while not done:
           # print("CCCCCCCCCCCCCCCCCCC", env.observation_space.shape)
           action = policy.act(ob)
           ob, reward, done, info = env.step(action)
           print("info =", info)
           event = info['event']
           if render:
               env.render('human')  # robocentric=True, save_to_file=True)
           env_time += env._get_dt()
           if isinstance(event, Danger):
               too_close += 1
               min_dist.append(event.min_dist)
           rewards.append(reward)
           states += 1
        
        
        #    episodic_info['deadlock'] += info['deadlock']
        #    if isinstance(event, Deadlock):
        #        deadlock.append(episodic_info['deadlock'])
           

        #    episodic_info['avg_speed'] += (info['speed'] - episodic_info['avg_speed']) / states
        #    episodic_info['speed_violation'] += (info['speed'] > 1)
        #    episodic_info['social_violation_cnt'] += info['social_violation_cnt']
        #    episodic_info['personal_violation_cnt'] += info['personal_violation_cnt']
        #    episodic_info['jerk_cost'] += info['jerk_cost']
        #    episodic_info['aggregated_time'] += info['aggregated_time']
        #    if episodic_info['side_preference'] is None:
        #        episodic_info['side_preference'] = info['side_preference']
           # elif info['side_preference'] is not None and episodic_info['side_preference'] != info['side_preference']:
           #     raise Exception('Side preference changed mid-episode') # Side preference calculation is not allowed to change mid-episode
 
        #    if isinstance(event, ReachGoal):
            #    avg_speed.append(episodic_info['avg_speed'])
            #    speed_violation.append(episodic_info['speed_violation'] / env.soadrl_sim.global_time)
            #    social_violation_cnt.append(episodic_info['social_violation_cnt'] / env.soadrl_sim.global_time)
            #    personal_violation_cnt.append(episodic_info['personal_violation_cnt'] / env.soadrl_sim.global_time)
            #    jerk_cost.append(episodic_info['jerk_cost'] / env.soadrl_sim.global_time)
            #    aggregated_time.append(episodic_info['aggregated_time'])
            #    side_preference.append(episodic_info['side_preference'])
            #    discomfort += episodic_info['discomfort']
    #    print("info = ", info)
    #    if info('collision_Pedestrian'):
    #        collision_pedestrian += 1
        #    collision_ped_times += 1
       if isinstance(event, ReachGoal):
           success += 1
           success_times.append(env_time)
       elif isinstance(event, ReachGoal_Ped):
           success_ped += 1
           success_ped_times.append(env_time)
       elif isinstance(event, Collision):
           collision += 1
           collision_cases.append(i)
           collision_times.append(env_time)
       elif isinstance(event, CollisionOtherAgent):
           collision_other_agent += 1
           collision_other_agent_cases.append(i)
           collision_other_agent_times.append(env_time)
       elif isinstance(event, Timeout):
           timeout += 1
           timeout_cases.append(i)
           timeout_times.append(env_time)
       elif isinstance(event, Deadlock):
           deadlock += 1
           deadlock_cases.append(i)
       else:
           raise ValueError('Invalid end signal from environment')
      
    
       cumulative_rewards.append(sum([pow(gamma, t * env.soadrl_sim.time_step * env.soadrl_sim.robot.v_pref)
                                      * reward for t, reward in enumerate(rewards)]))
 
#    left_percentage = side_preference.count(0) / len(side_preference) if len(side_preference) > 0 else 0
#    right_percentage = side_preference.count(1) / len(side_preference) if len(side_preference) > 0 else 0
 
 
 
#    success_rate = success / float(num_episodes)
#    collision_rate = collision / float(num_episodes)
#    collision_other_agent_rate = collision_other_agent / \
#        float(num_episodes)
#    timeout_rate = timeout / float(num_episodes)
 
#    assert success + collision + timeout + collision_other_agent == num_episodes
#    avg_nav_time = sum(success_times) / float(len(success_times)
#                                              ) if success_times else np.nan

  
#    print('%d were collisions with pedestrians out of %d collisions', collision_pedestrian, collision)
#    print('social violation: {:.2f}+-{:.2f}'.format( np.mean(social_violation_cnt), np.std(social_violation_cnt)))
#    print('personal violation: {:.2f}+-{:.2f}'.format( np.mean(personal_violation_cnt), np.std(personal_violation_cnt)))
#    print('jerk cost: {:.2f}+-{:.2f}'.format(np.mean(jerk_cost),np.std(jerk_cost)))
#    print('aggregated time: {:.2f}+-{:.2f}'.format(np.mean(aggregated_time),np.std(aggregated_time)))
#    print('speed: {:.2f}+-{:.2f}'.format(np.mean(avg_speed),np.std(avg_speed)))
#    print('speed violation : {:.2f}+-{:.2f} '.format(np.mean(speed_violation),np.std(speed_violation)))
#    print('left %: {:.2f}'.format(left_percentage))
#    print('right %: {:.2f}'.format(right_percentage))

   config_dir = resource_filename('crowd_nav', 'config')
   config_file = os.path.join(config_dir, 'test_soadrl_static.config')
   config_file = os.path.expanduser(config_file)
   config = configparser.RawConfigParser()
   config.read(config_file)

#    env = gym.make('CrowdSim-v0')
#    robot = Robot(config, 'humans')
#    env.set_robot(robot)
#    while not done:
#             action = robot.act(ob)
#             ob, _, done, info = env.step(action)
#             print("info = ", info)
        #     if info == 'Reach Goal':
        #         human_times = env.get_human_times()

        #         for time in human_times:
        # #    print('Human time is: %.2f', time)
        #             #    print("Average time for humans to reach goal: %.2f', sum(human_times)/len(human_times))")
        #             logging.info('Human time is: %.2f', time)
        #             logging.info('Average time for humans to reach goal: %.2f', sum(human_times)/len(human_times))
#    print(
#        """has success rate: {:.2f}, collision rate: {:.2f},
#        timeout rate: {:.2f}, nav time: {:.2f}, total reward: {:.4f}""".format(
#            success_rate,
#            collision_rate,
#            timeout_rate,
#            avg_nav_time,
#            np.mean(cumulative_rewards)
#        )
#    )

   ## FOR PEDESTRIANS WITHOUT ROBOT
   ped_avg_nav_time = sum(success_ped_times) / float(len(success_ped_times)
                                        ) if success_ped_times else np.nan

   avg_deadlock_time = sum(deadlock) / (len(deadlock)
                                             ) if deadlock else np.nan
  
#   computational_time = total_time / number of episodes
#    print("env.soadrl_sim.global_time = ", env.soadrl_sim.global_time)
   print("time.time() = ", time.time())
   print("start_time() = ", start_time)

   total_time = (time.time() - start_time)
   print("total_time = ", total_time)
   computational_time = total_time / float (num_episodes)

   print(
       """has Average deadlock rate: {:.2f}, Computational time: {:.2f},
       Pedestrian navigation time: {:.2f}""".format(
           avg_deadlock_time,
           computational_time,
           ped_avg_nav_time,
       )
   )

#    total_time = sum(success_times + collision_times + collision_other_agent_times + timeout_times)
#    print(
#        'Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
#        too_close / float(total_time),
#        np.mean(min_dist))
 
#    if print_failure:
#        print('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
#        print('Collision from other agent cases: ' + ' '.join([str(x) for x in collision_other_agent_cases]))
#        print('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))
 
   return avg_deadlock_time, computational_time, ped_avg_nav_time#success_rate, avg_nav_time, 
 
 
if __name__ == '__main__':
   args, _ = parse_common_args()
 
   if args.environment is None or args.environment == "navreptrain":
       env = NavRepTrainEncodedEnv(args.backend, args.encoding, silent=False, scenario='test')
       policy = NavRepCPolicy()
   else:
       raise NotImplementedError
 
   run_test_episodes(env, policy, render=args.render)
 













