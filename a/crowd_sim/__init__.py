from gym.envs.registration import register

register(
    id='CrowdSim-v1',
    entry_point='a.crowd_sim.envs:CrowdSim2',
)


