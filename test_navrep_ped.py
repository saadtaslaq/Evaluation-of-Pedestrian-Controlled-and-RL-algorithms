######################################## KAREN ZHIXIAN CHANGES ######################################
# import os
# import numpy as np
# from tqdm import tqdm
# from stable_baselines import PPO2
# from navrep.envs.navreptrainencodedenv import NavRepTrainEncodedEnv
# import time
# from crowd_sim.envs.utils.info import Timeout, ReachGoal, Danger, Collision, CollisionOtherAgent
# from navrep.tools.commonargs import parse_common_args

# import crowd_sim  # adds CrowdSim-v0 to gym  # noqa
# from crowd_sim.envs.crowd_sim import CrowdSim  # reference to env code  # noqa

# from crowd_sim.envs.crowd_sim import ReachGoal 

# class NavRepCPolicy(object):
#     """ wrapper for gym policies """
#     def __init__(self, model=None): 
#         if model is not None:
#             self.model = model
#         else:
#             self.model_path = os.path.expanduser(
#                 # "~/navrep/models/gym/navreptrainencodedenv_2020_09_17__09_15_17_PPO_VAE_LSTM_V_ONLY_V32M512_ckpt")
#                 # "~/navrep/models/gym/navreptrainencodedenv_2020_09_22__01_14_06_PPO_VAE_LSTM_V_ONLY_V32M512_ckpt")
#                 # "~/navrep/models/gym/navreptrainencodedenv_2020_09_19__13_27_16_PPO_VAE_LSTM_V_ONLY_V32M512_ckpt")
#                 # "~/navrep/models/gym/navreptrainencodedenv_2022_05_17__19_30_10_PPO_VAE_LSTM_V_ONLY_V32M512_ckpt")
#                 # "~/navrep/models/gym/navreptrainencodedenv_2022_01_31__16_23_55_PPO_VAE_LSTM_V_ONLY_V32M512_ckpt")
#                 "~/navrep/models/gym/navreptrainencodedenv_2022_05_17__23_59_07_PPO_VAE_LSTM_V_ONLY_V32M512_ckpt")
                
                
#             self.model = PPO2.load(self.model_path) 
#             print("Model '{}' loaded".format(self.model_path))

#     def act(self, obs):
#         action, _states = self.model.predict(obs, deterministic=True)
#         return action


# def run_test_episodes(env, policy, render=False, print_failure=True, num_episodes=100):
#     success_times = []
#     collision_times = []
#     ped_collision_times = []
#     collision_other_agent_times = []
#     timeout_times = []
#     ped_timeout_times = []

#     success = 0
#     ped_success=0
#     collision = 0
#     ped_collision=0
#     collision_other_agent = 0
#     ped_collision_other_agent=0
#     timeout = 0
#     too_close = 0
#     min_dist = []
#     ped_success_times = []
#     cumulative_rewards = []
#     collision_cases = []
#     collision_other_agent_cases = []
#     ped_collision_cases = []
#     timeout_cases = []
#     deadlock = 0
#     progress_bar = tqdm(range(num_episodes), total=num_episodes)
#     start_time = time.time()
    
#     start_times =[]
#     end_times = []


#     deadlock_per_episode = []
#     for i in progress_bar:
#         check = 0
#         progress_bar.set_description("Case {}".format(i))
#         ob = env.reset()
#         done = False
#         env_time = 0
#         deadlock = 0
#         while not done: # False
#             print("done=", done)
#             action = policy.act(ob)
#             ob, _, done, info = env.step(action)
#             event = info['event']
#             if render:
#                 env.render('human')  # robocentric=True, save_to_file=True)
#             env_time += env._get_dt()
#             print("event = ", event)
#             print("info = ", info)

#             if isinstance(event, Danger):
#                 too_close += 1
#                 deadlock += 1
#                 min_dist.append(event.min_dist)
#                 check = 1
#             elif isinstance(event, ReachGoal):
#                 print("YES REACHED GOAL")
#                 # ped_success += 1
#                 # ped_success_times.append(env_time)
#                 success += 1
#                 success_times.append(env_time)
#         if check ==1:
#             deadlock_per_episode.append(deadlock)
#         else:
#             deadlock_per_episode.append(0)
#         env_time += env._get_dt()
#         print("envv timeeeee: ", env_time)
#         if isinstance(event, Collision):
#             collision += 1
#             collision_cases.append(i)
#             collision_times.append(env_time)
#         elif isinstance(event, CollisionOtherAgent):
#             collision_other_agent += 1
#             collision_other_agent_cases.append(i)
#             collision_other_agent_times.append(env_time)
#         elif isinstance(event, Timeout):
#             timeout += 1
#             timeout_cases.append(i)
#             timeout_times.append(env_time)
#         # /home/caris/Code/navrep/navrep/scripts/test_navrep.py
#         # else:
#         #     raise ValueError('Invalid end signal from environment')
#         #     TODO: for each in Pedstrain
#         #             if isinstance(event, ReachGoal):
#         #                 success += 1
#         #                 success_times.append(env_time)
#         #             elif isinstance(event, Collision):
#         #                 collision += 1
#         #                 collision_cases.append(i)
#         #                 collision_times.append(env_time)
#         #             elif isinstance(event, CollisionOtherAgent):
#         #                 collision_other_agent += 1
#         #                 collision_other_agent_cases.append(i)
#         #                 collision_other_agent_times.append(env_time)
#         #             elif isinstance(event, Timeout):
#         #                 timeout += 1
#         #                 timeout_cases.append(i)
#         #                 timeout_times.append(env_time)
#         end_time = time.time() - start_time
#         end_times.append(end_time)

    
#     # print(end_times)


#     # ##### Statistic for all Peds with the appearance of the robot #######
 
#     # ped_success_rate = ped_success/ float(num_episodes)
#     # print("ped_success = ", ped_success)
#     # ped_collision_rate = ped_collision / float(num_episodes)
#     # ped_collision_other_agent_rate = ped_collision_other_agent / \
#     #     float(num_episodes)


#     # ped_avg_nav_time = sum(ped_success_times) / float(len(ped_success_times)
#     #                                     ) if ped_success_times else np.nan
    
#     # total_time = sum(ped_success_times + ped_collision_times + collision_other_agent_times + ped_timeout_times)
#     # deadlocks = too_close / float(total_time)

#     # total_time_ped = (time.time() - start_time)
#     # print(total_time_ped)
#     # computational_time = total_time_ped / float(num_episodes)

#     # print("""Ped has success rate: {:.2f} Ped nav time: {:.2f}, 
#     #         computation time: {:.4f}, deadlocks : {:.4f},
#     #         Ped colliding with other agents rate: {:.4f}""".format(
#     #         ped_success_rate,
#     #         ped_avg_nav_time,
#     #         computational_time,
#     #         deadlocks,
#     #         ped_collision_other_agent_rate
#     #     )
#     # )                                         ) if success_times else np.nan
 
#                                 ##### Statistic for all Peds with the appearance of the robot #######
 
#     # success_rate = success/ float(num_episodes)
#     # print("success = ", success)
#     # collision_rate = collision / float(num_episodes)
#     # collision_other_agent_rate = collision_other_agent / \
#     #     float(num_episodes)


#     # avg_nav_time = sum(success_times) / float(len(success_times)
#     #                                     ) if success_times else np.nan

#     # statistic for Robot
#     success_rate = success / float(num_episodes)
#     collision_rate = collision / float(num_episodes)
#     collision_other_agent_rate = collision_other_agent / \
#         float(num_episodes)
#     assert success + collision + timeout + collision_other_agent == num_episodes
#     # if success_times == 0:
#     #     print('Success times: 0')
#     # else:   
#     avg_nav_time = sum(success_times) / float(len(success_times)
#                                                 ) if success_times else np.nan

    
#     total_time = sum(success_times + collision_times + collision_other_agent_times + timeout_times)
#     # print("totaltime:", total_time)
#     # print("tooclose ", too_close)
#     # print("success_times", success_times)
#     # print("collision_times", collision_times)
#     # print("collision_other_agent_times", collision_other_agent_times)
#     # print("timeout_times", timeout_times)
#     # print('collision case ', collision_cases)
#     print('deadlock per episode is: ', deadlock_per_episode)
#     deadlocks = too_close / float(total_time)

#     total_time_ped = (time.time() - start_time)
#     # print('total_time_ped is', total_time_ped)
#     computational_time = total_time_ped / float(num_episodes)

#     print("""Robot has success rate: {:.2f} Robot nav time: {:.2f}, 
#             computation time: {:.4f},
#             mean of deadlock: {:.4f},
#             sd of deadlock: {:.4f}""".format(
#             success_rate,
#             total_time_ped,
#             computational_time,
#             np.mean(deadlock_per_episode),
#             np.std(deadlock_per_episode)

#         )
#     )
   

# #     print(
# #         """has success rate: {:.2f}, collision rate: {:.2f},
# #         collision from other agents rate: {:.2f}, nav time: {:.2f}, total reward: {:.4f}""".format(
# #             success_rate,
# #             collision_rate,
# #             collision_other_agent_rate,
# #             avg_nav_time,
# #             np.mean(cumulative_rewards)
# #         )
# #     )computational_timeverage min separate distance in danger: %self.map_list.2f',
# #         too_close / float(total_time),
# #         np.mean(min_dist))


# #     if print_failure:
# #         print('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
# #         print('Collision fro    total_time = sum(success_times + collision_times + collision_other_agent_times + timeout_times)
# # m other agent cases: ' + ' '.join([str(x) for x in collision_other_agent_cases]))
# #         print('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))

#     return success_rate, avg_nav_time, computational_time, deadlocks, collision_other_agent_rate


# if __name__ == '__main__':
#     args, _ = parse_common_args()

#     if args.environment is None or args.environment == "navreptrain":
#         env = NavRepTrainEncodedEnv(args.backend, args.encoding, silent=True, scenario='test')
#         policy = NavRepCPolicy()
#     else:
#         raise NotImplementedError

#     run_test_episodes(env, policy, render=args.render)

############################################## SAAD CHANGES #####################################
import os
import numpy as np
from tqdm import tqdm
from stable_baselines import PPO2
from navrep.envs.navreptrainencodedenv import NavRepTrainEncodedEnv
from navrep.envs.navreptrainenv import NavRepTrainEnv

# from crowd_sim.envs.utils.information import Timeout, ReachGoal, Danger, Collision, CollisionOtherAgent
# from crowd_sim.envs.utils.information import Deadlock, ReachGoal_Ped
from crowd_sim.envs.utils.info import *
from navrep.tools.commonargs import parse_common_args
from crowd_sim.envs.crowd_sim import CrowdSim  # reference to env code  # noqa
# from crowd_sim.envs.utils.info import 


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
 
 

def run_test_episodes(env, policy, render=False, print_failure=True, num_episodes=100):
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
   deadlock_count = []
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
   start_time = time.time()

   for i in progress_bar:
       print("i", i)

       progress_bar.set_description("Case {}".format(i))

       ob = env.reset()
       done = False
       env_time = 0
       rewards = []
       Check = 0
 
 
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
       deadlock = 0

       while not done:
           # print("CCCCCCCCCCCCCCCCCCC", env.observation_space.shape)
           action = policy.act(ob)
           ob, reward, done, info, distance_to_goal = env.step(action)
           print("distance_to_goal fgdfdgdfgfdg", distance_to_goal)
           
           print("info =", info)
           event = info['event']
           print("event ==== ", event)
           if render:
               env.render('human')  # robocentric=True, save_to_file=True)
           env_time += env._get_dt()
           if isinstance(event, Danger):
               deadlock += 1
               min_dist.append(event.min_dist)
               
               Check = 1
        #    rewards.append(reward)
        #    states += 1
           elif isinstance(event, ReachGoal):
            #    print("YES REACHED GOAL")
               success += 1
               success_times.append(env_time)
       if Check == 1:
           deadlock_count.append(deadlock)
       else:
            deadlock_count.append(0)

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
    #    elif isinstance(event, ReachGoal_Ped):
    #        success_ped += 1
    #        success_ped_times.append(env_time)
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
    #    elif isinstance(event, Deadlock):
    #        deadlock += 1
    #        deadlock_cases.append(i)
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

#    config_dir = resource_filename('crowd_nav', 'config')
#    config_file = os.path.join(config_dir, 'test_soadrl_static.config')
#    config_file = os.path.expanduser(config_file)
#    config = configparser.RawConfigParser()
#    config.read(config_file)

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
#    ped_avg_nav_time = sum(success_ped_times) / float(len(success_ped_times)
#                                         ) if success_ped_times else np.nan

#    print("deadlock = ", deadlock)
# 
#    avg_deadlock_timesum(deadlock_cases) / (len(deadlock_cases)
#                                              ) if deadlock_cases else np.nan
  
#   computational_time = total_time / number of episodes
#    print("env.soadrl_sim.global_time = ", env.soadrl_sim.global_time)
#    print("time.time() = ", time.time())
#    print("start_time() = ", start_time)

   total_time = (time.time() - start_time)
#    print("total_time = ", total_time)
   computational_time = total_time / float (num_episodes)
   ped_avg_nav_time = sum(success_times) / float(len(success_times)
                                            ) if success_times else np.nan
                                            
   ped_success_rate = success / float((num_episodes)
                                            ) if success_times else np.nan
   print("ped_success_rate = ",ped_success_rate)
   total_time_ped = sum(success_times + collision_times + collision_other_agent_times + timeout_times)
   avg_deadlock_time = deadlock / float(total_time_ped)
   print("computational_time = ", computational_time)
   print("ped_avg_nav_time = ", ped_avg_nav_time)
   print("ped_success_rate = ", ped_success_rate)
   print("avg_deadlock_time = ", avg_deadlock_time)
   print("deadlock_count = ", deadlock_count)
   deadlock_counter = 0
    # for i in range(len(deadlock_count)):
    #     print(deadlock_count[i])
   for i in range(len(deadlock_count)):
       deadlock_counter += deadlock_count[i]
#    print("SUCCESS = ", success)
#    print("success_times = ", success_times)
#    print("too_close = ", deadlock)
   maps_with_deadlock =[]
   for i in deadlock_count:
       if i > 0:
           maps_with_deadlock.append(i+1)

   
   import matplotlib.pyplot as plt
   plt.figure(1) 
   bar_width = 1 # set this to whatever you want
   positions = np.arange(100)
   plt.bar(positions, deadlock_count, bar_width)
# #    plt.xticks(positions + bar_width / 2, ('0', '1', '2', '3'))
   plt.show()

#    plt.figure(2)
#    d_positions = np.arange(100)
#    plt.bar(d_positions, distance_to_goal[0:100], bar_width)
# #    plt.xticks(positions + bar_width / 2, ('0', '1', '2', '3'))
#    plt.show()

   print("maps_with_deadlock: ", maps_with_deadlock)
   print(
       """Frequency of being in deadlock: {:.2f}, Deadlock for pedestrian: {:.2f}, Standard deviation of Deadlocks {:.2f} 
       Computational time: {:.2f}, Pedestrian navigation time: {:.2f}, Pedestrian success rate: {:.2f}""".format(
           avg_deadlock_time,
           np.sum(deadlock_count),

           np.std(deadlock_count),
           computational_time,
           ped_avg_nav_time,
           ped_success_rate)
   )
    
#         )
   

  

   
#    total_time = sum(success_times) #+ timeout_times)
#    print("too_close = ", too_close)
#    print("min_dist = ", min_dist)
#    print(
#        """Frequency of being in deadlock: {:.2f} and average min separate distance in deadlock: {:.2f}""",
#        too_close / float(total_time),
#        np.mean(min_dist))
#    total_time = sum(success_times + collision_times + collision_other_agent_times + timeout_times)
#    print(
#        'Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
#        too_close / float(total_time),
#        np.mean(min_dist))
 
   if print_failure:
       print('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
       print('Collision from other agent cases: ' + ' '.join([str(x) for x in collision_other_agent_cases]))
       print('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))
 
   return avg_deadlock_time, computational_time, ped_avg_nav_time, ped_success_rate #m success_rate, avg_nav_time, 
 
 
if __name__ == '__main__':
   args, _ = parse_common_args()
 
   if args.environment is None or args.environment == "navreptrain":
       env = NavRepTrainEncodedEnv(args.backend, args.encoding, silent=False, scenario='test')
       policy = NavRepCPolicy()
   else:
       raise NotImplementedError
   run_test_episodes(env, policy, render=args.render)
