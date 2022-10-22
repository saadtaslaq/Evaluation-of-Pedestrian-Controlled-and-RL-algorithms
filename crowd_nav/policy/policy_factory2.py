from crowd_sim.envs.policy.orca import ORCA
from crowd_nav.policy.network_om import SDOADRL

policy_factory = dict()

policy_factory['orca'] = ORCA
policy_factory['sdoadrl'] = SDOADRL
