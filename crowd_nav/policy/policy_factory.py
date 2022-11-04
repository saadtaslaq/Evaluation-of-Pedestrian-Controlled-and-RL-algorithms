from crowd_sim.envs.policy.policy import Policy
from crowd_nav.policy.cadrl_original_data import CADRL_ORIGINAL
from crowd_nav.policy.sarl import SARL
from crowd_sim.envs.policy.orca import ORCA
from crowd_nav.policy.network_om import SDOADRL
from crowd_sim.envs.policy.random_policy import RandomPolicy
from crowd_sim.envs.policy.social_force import SOCIAL_FORCE
from crowd_sim.envs.policy.pysf import PYSF 
from crowd_sim.envs.policy.linear import Linear

def none_policy():
    return None

policy_factory = dict()

policy_factory['cadrl_original'] = CADRL_ORIGINAL
policy_factory['sarl'] = SARL
policy_factory['orca'] = ORCA
policy_factory['sdoadrl'] = SDOADRL
policy_factory['random'] = RandomPolicy
policy_factory['social_forces'] = SOCIAL_FORCE
policy_factory['pysf'] = PYSF
policy_factory['linear'] = Linear
policy_factory['none'] = none_policy
