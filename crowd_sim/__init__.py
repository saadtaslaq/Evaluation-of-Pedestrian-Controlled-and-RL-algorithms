from gym.envs.registration import register

register(
    id='CrowdSim-v0',
    entry_point='crowd_sim.envs:CrowdSim',
)

register(
    id='CrowdSimDict-v0',
    entry_point='crowd_sim.envs:CrowdSimDict',

)

register(
    id='NavRepTrainEnv-v0',
    entry_point='crowd_sim.envs:NavRepTrainEnv',
)

#register(
#    id='CrowdSim-v1',
#    entry_point='crowd_sim.envs:CrowdSim2',
#)

register(
    id='IANEnv-v0',
    entry_point='crowd_sim.envs:IANEnv',
)