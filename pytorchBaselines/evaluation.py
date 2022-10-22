from sre_constants import SUCCESS
import numpy as np
import torch
from time import sleep
from crowd_sim.envs.utils.info import *
from pytorchBaselines.a2c_ppo_acktr import utils
from a.crowd_sim.envs.utils.info import Collision, CollisionOtherAgent, ReachGoal, Timeout
from a.crowd_nav.policy.policy_factory import policy_factory
# from crowd_nav.policy.social_force import SOCIAL_FORCE
# from a.crowd_sim.envs.utils.human import Human


def evaluate(actor_critic, ob_rms, eval_envs, num_processes, device, config, logging, visualize=False,
             recurrent_type='GRU'):
    test_size = config.env.test_size

    if ob_rms:
        vec_norm = utils.get_vec_normalize(eval_envs)
        if vec_norm is not None:
            vec_norm.eval()
            vec_norm.ob_rms = ob_rms

    eval_episode_rewards = []

    if recurrent_type == 'LSTM':
        rnn_factor = 2
    else:
        rnn_factor = 1


    eval_recurrent_hidden_states = {}

    node_num = 1
    edge_num = actor_critic.base.human_num + 1
    eval_recurrent_hidden_states['human_node_rnn'] = torch.zeros(num_processes, node_num, config.SRNN.human_node_rnn_size * rnn_factor,
                                                                 device=device)

    eval_recurrent_hidden_states['human_human_edge_rnn'] = torch.zeros(num_processes, edge_num,
                                                                       config.SRNN.human_human_edge_rnn_size*rnn_factor,
                                                                       device=device)

    eval_masks = torch.zeros(num_processes, 1, device=device)

    success_times = []
    collision_times = []
    timeout_times = []
    path_lengths = []
    chc_total = []
    success = 0
    human_policy = 0
    collision = 0
    timeout = 0
    too_close = 0.
    min_dist = []
    cumulative_rewards = []
    collision_cases = []
    timeout_cases = []
    success_cases = []
    gamma = 0.9
    baseEnv = eval_envs.venv.envs[0].env

    avg_speed = []
    speed_violation = []
    social_violation_cnt = []
    personal_violation_cnt = []
    jerk_cost = []
    aggregated_time = []
    side_preference = []
    discomfort = 0

    collision_pedestrian = 0
    obs = eval_envs.reset()
    for k in range(test_size):
        done = False
        rewards = []
        stepCounter = 0
        episode_rew = 0

        global_time = 0.0
        path = 0.0
        chc = 0.0

        last_pos = obs['robot_node'][0, 0, 0:2].cpu().numpy()  # robot px, py
        last_angle = np.arctan2(obs['temporal_edges'][0, 0, 1].cpu().numpy(), obs['temporal_edges'][0, 0, 0].cpu().numpy())  # robot theta

        episodic_info = {
            'discomfort' : 0,
            'min_dist' : [],
            'end_event' : None,

            'avg_speed' : 0,
            'speed_violation' : 0,
            'social_violation_cnt' : 0,
            'personal_violation_cnt' : 0,
            'jerk_cost' : 0,
            'aggregated_time' : 0,
            'side_preference' : None
        }
        states = 0

        while not done:
            # sleep(0.035)
            stepCounter = stepCounter + 1
            with torch.no_grad():

                # ob, reward, done, info, distance_to_goal = env.step(action)
                _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                    obs,
                    eval_recurrent_hidden_states,
                    eval_masks,
                    deterministic=True)
            if not done:
                global_time = baseEnv.soadrl_sim.global_time
            if visualize:
                eval_envs.render()
            states += 1

            # Obser reward and next obs
            obs, rew, done, infos = eval_envs.step(action)
            episodic_info['avg_speed'] += (infos[0]['speed'] - episodic_info['avg_speed']) / states
            episodic_info['speed_violation'] += (infos[0]['speed'] > 1)
            episodic_info['social_violation_cnt'] += infos[0]['social_violation_cnt']
            episodic_info['personal_violation_cnt'] += infos[0]['personal_violation_cnt']
            episodic_info['jerk_cost'] += infos[0]['jerk_cost']
            episodic_info['aggregated_time'] += infos[0]['aggregated_time']
            if episodic_info['side_preference'] is None:
                episodic_info['side_preference'] = infos[0]['side_preference']
            # elif infos[0]['side_preference'] is not None and episodic_info['side_preference'] != infos[0]['side_preference']:
            #     raise Exception('Side preference changed mid-episode') # Side preference calculation is not allowed to change mid-episode

            path = path + np.linalg.norm(np.array([last_pos[0] - obs['robot_node'][0, 0, 1].cpu().numpy(),
                                                   last_pos[1] - obs['robot_node'][0, 0, 2].cpu().numpy()]))

            cur_angle = np.arctan2(obs['temporal_edges'][0, 0, 1].cpu().numpy(), obs['temporal_edges'][0, 0, 0].cpu().numpy())
            chc = chc +  abs(cur_angle - last_angle)

            last_pos = obs['robot_node'][0, 0, 0:2].cpu().numpy()  # robot px, py
            last_angle = cur_angle

            
            rewards.append(rew)


            if isinstance(infos[0]['event'], Danger):
                too_close = too_close + 1
                min_dist.append(infos[0]['event'].min_dist)

            if isinstance(infos[0]['event'], ReachGoal):
                avg_speed.append(episodic_info['avg_speed'])
                speed_violation.append(episodic_info['speed_violation'] / global_time)
                print("social violation",global_time)
                social_violation_cnt.append(episodic_info['social_violation_cnt'] / global_time)
                personal_violation_cnt.append(episodic_info['personal_violation_cnt'] / global_time)
                jerk_cost.append(episodic_info['jerk_cost'] / global_time)
                aggregated_time.append(episodic_info['aggregated_time'])
                side_preference.append(episodic_info['side_preference'])
                discomfort += episodic_info['discomfort']

            episode_rew += rew[0]


            eval_masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device)

            for info in infos:
                if 'episode' in info.keys():
                    eval_episode_rewards.append(info['episode']['r'])

            
        if infos[0]['collision_pedestrian']:
            collision_pedestrian += 1


        print('')
        print('Reward={}'.format(episode_rew))
        print('Episode', k, 'ends in', stepCounter)
        path_lengths.append(path)
        chc_total.append(chc)

        print('thing', infos[0]['event'], Timeout)

        if isinstance(infos[0]['event'], ReachGoal):
            success += 1
            success_cases.append(k)
            success_times.append(global_time)
            print('Success')
        elif isinstance(infos[0]['event'], Collision) or isinstance(infos[0]['event'], CollisionOtherAgent):
            collision += 1
            collision_cases.append(k)
            collision_times.append(global_time)
            print('Collision')
        elif isinstance(infos[0]['event'], Timeout):
            timeout += 1
            timeout_cases.append(k)
            timeout_times.append(baseEnv.time_limit)
            print('Time out')
        else:
            raise ValueError('Invalid end signal from environment')

        cumulative_rewards.append(sum([pow(gamma, t * baseEnv.soadrl_sim.time_step * baseEnv.soadrl_sim.robot.v_pref)
                                       * reward for t, reward in enumerate(rewards)]))
    np.savetxt('vx', np.array(baseEnv.soadrl_sim.vx_arr))
    left_percentage = side_preference.count(0) / len(side_preference) if len(side_preference) > 0 else 0
    right_percentage = side_preference.count(1) / len(side_preference) if len(side_preference) > 0 else 0
    logging.info(
    f'social violation: {np.mean(social_violation_cnt):.2f}+-{np.std(social_violation_cnt):.2f}, '
    f'personal violation: {np.mean(personal_violation_cnt):.2f}+-{np.std(personal_violation_cnt):.2f}, '
    f'jerk cost: {np.mean(jerk_cost):.2f}+-{np.std(jerk_cost):.2f}, '
    f'aggregated time: {np.mean(aggregated_time):.2f}+-{np.std(aggregated_time):.2f}, '
    f'speed: {np.mean(avg_speed):.2f}+-{np.std(avg_speed):.2f}, '
    f'speed violation: {np.mean(speed_violation):.2f}+-{np.std(speed_violation):.2f}, '
    f'left %: {left_percentage:.2f}, '
    f'right %: {right_percentage:.2f}')

    logging.info('%d were collisions with pedestrians out of %d collisions', collision_pedestrian, collision)

    success_rate = success / test_size
    collision_rate = collision / test_size
    timeout_rate = timeout / test_size
    assert success + collision + timeout == test_size
    # print('success', success_times)
    # print('limit', baseEnv.soadrl_sim.time_limit)
    avg_nav_time = sum(success_times) / len(
        success_times) if success_times else baseEnv.soadrl_sim.time_limit  # baseEnv.env.time_limit

    extra_info = ''
    phase = 'test'
    logging.info(
        '{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, '
        'timeout rate: {:.2f}, average time: {:.2f}, total reward: {:.4f}'.
            format(phase.upper(), extra_info, success_rate, collision_rate, timeout_rate, avg_nav_time,
                   np.average(cumulative_rewards)))
    if phase in ['val', 'test']:
        total_time = sum(success_times + collision_times)#+ timeout_times)
        if min_dist:
            avg_min_dist = np.average(min_dist)
        else:
            avg_min_dist = float("nan")
        logging.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
                     too_close * baseEnv.soadrl_sim.time_step / total_time, avg_min_dist)

    logging.info(
        '{:<5} {}has average path length: {:.2f}, CHC: {:.2f}'.
            format(phase.upper(), extra_info, sum(path_lengths) / test_size, sum(chc_total) / test_size))
    logging.info('Success cases: ' + ' '.join([str(x) for x in success_cases]))
    logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
    logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))

    print("Success Rate: ", success_rate)
    print("Collision Rate: ",collision_rate)
    print("Timeout Rate: ", timeout_rate)
    eval_envs.close()

    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))


    
