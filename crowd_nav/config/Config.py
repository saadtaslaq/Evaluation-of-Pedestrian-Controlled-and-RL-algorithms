import numpy as np


class BaseConfig(object):
    def __init__(self):
        pass


class Config(object):
    test = BaseConfig()
    test.social_metrics = False

    sim = BaseConfig()
    sim.render = False  # show GUI for visualization
    # currently selected at random
    # select which scenarios you would like to train
    sim.train_val_sim = [
        "circle_crossing",
        "square_crossing",
        "parallel_traffic",
        "perpendicular_traffic",
    ]
    # select which scenarios you would like to test
    # NOTE: use ["side_pref_passing", "side_pref_overtaking", "side_pref_crossing"] IN A SINGLE ELEM LIST to test side_preference
    sim.test_sim = [
        "circle_crossing",
        "square_crossing",
        "parallel_traffic",
        "perpendicular_traffic",
        # "side_pref_passing",
        # "side_pref_overtaking",
        # "side_pref_crossing",
    ]
    sim.square_width = 20

    # automatically infer based on selected scenario
    test.side_preference = any("side_pref" in s_ for s_ in sim.test_sim)

    sim.circle_radius = 6 if not test.social_metrics and not test.side_preference else 4
    sim.human_num = 5 if not test.side_preference else 1
    # Group environment: set to true; FoV environment: false
    sim.group_human = False

    env = BaseConfig()
    env.env_name = "CrowdSimDict-v0"  # name of the environment
    env.time_limit = 50
    env.time_step = 0.25
    env.val_size = 100
    env.test_size = 500

    if test.social_metrics:
        env.test_size = 2000
    elif test.side_preference:
        env.test_size = 200

    env.randomize_attributes = True
    env.seed = 0  # env random seed

    reward = BaseConfig()
    reward.time_factor = False
    reward.normalize = False
    reward.potential_based = True
    reward.exponential = False
    reward.norm_zones = False
    assert reward.potential_based != reward.exponential
    reward.success_reward = 10 if not reward.normalize else 1
    reward.collision_penalty = -20 if not reward.normalize else -1
    reward.timeout_penalty = -20 if not reward.normalize else -1  # (UNUSED)
    # discomfort distance for the front half of the robot (UNUSED)
    reward.discomfort_dist_front = 0.25
    # discomfort distance for the back half of the robot
    reward.discomfort_dist_back = 0.25
    reward.discomfort_penalty_factor = 10 if not reward.normalize else 0.5
    reward.discomfort_penalty_factor *= env.time_step

    reward.potential_factor = 2 if not reward.normalize else 0.1
    reward.exp_factor = 0.5 if not reward.normalize else 0.025
    reward.exp_denom = 6  # set to same as sim.circle_radius
    reward.gamma = 0.99  # discount factor for rewards
    # from SA-CADRL
    reward.norm_zone_side = 'lhs'  # 'lhs' or 'rhs'
    reward.norm_zone_penalty = -0.5

    humans = BaseConfig()
    humans.visible = True  # a human is visible to other humans and the robot
    # orca or social_force for now
    humans.policy = "social_force"
    humans.radius = 0.3
    humans.v_pref = 1
    humans.sensor = "coordinates"
    # FOV = this values * PI
    humans.FOV = 2.0

    # a human may change its goal before it reaches its old goal
    humans.random_goal_changing = True if not test.side_preference else False
    humans.goal_change_chance = 0.25

    # a human may change its goal after it reaches its old goal
    humans.end_goal_changing = True if not test.side_preference else False
    humans.end_goal_change_chance = 1.0

    # a human may change its radius and/or v_pref after it reaches its current goal
    humans.random_radii = False
    humans.random_v_pref = False

    # one human may have a random chance to be blind to other agents at every time step
    humans.random_unobservability = False
    humans.unobservable_chance = 0.3

    humans.random_policy_changing = False

    robot = BaseConfig()
    robot.visible = False  # the robot is visible to humans
    # srnn for now
    robot.policy = "srnn"
    # robot.policy = 'convgru'

    robot.radius = 0.3
    robot.v_pref = 1
    robot.sensor = "coordinates"
    # FOV = this values * PI
    robot.FOV = 2.0

    noise = BaseConfig()
    noise.add_noise = False
    # uniform, gaussian
    noise.type = "uniform"
    noise.magnitude = 0.1

    lidar = BaseConfig()
    # set robot_radius = robot.radius BELOW
    lidar.enable = False
    lidar.viz = False
    lidar.cfg = {"max_range": 5, "num_beams": 180, "robot_radius": robot.radius}

    action_space = BaseConfig()
    # holonomic or unicycle
    action_space.kinematics = "holonomic"

    # config for ORCA
    orca = BaseConfig()
    orca.neighbor_dist = 10
    orca.safety_space = 0.15
    orca.time_horizon = 5
    orca.time_horizon_obst = 5

    # social force
    sf = BaseConfig()
    sf.A = 2.0
    sf.B = 1
    sf.KI = 1

    social = BaseConfig()
    social.min_personal_space = 0.2
    social.max_walking_speed = 1.5

    # ppo
    ppo = BaseConfig()
    ppo.num_mini_batch = 2  # number of batches for ppo
    ppo.num_steps = 30  # number of forward steps
    ppo.recurrent_policy = True  # use a recurrent policy
    ppo.epoch = 5  # number of ppo epochs
    ppo.clip_param = 0.2  # ppo clip parameter
    ppo.value_loss_coef = 0.5  # value loss coefficient
    ppo.entropy_coef = 0.0  # entropy term coefficient
    ppo.use_gae = True  # use generalized advantage estimation
    ppo.gae_lambda = 0.95  # gae lambda parameter

    # ConvGRU config
    ConvGRU = BaseConfig()
    ConvGRU.input_size = 256
    ConvGRU.hidden_size = 256

    # SRNN config
    SRNN = BaseConfig()
    # RNN size
    SRNN.human_node_rnn_size = 128  # Size of Human Node RNN hidden state
    SRNN.human_human_edge_rnn_size = 256  # Size of Human Human Edge RNN hidden state

    # Input and output size
    SRNN.human_node_input_size = 3  # Dimension of the node features
    SRNN.human_human_edge_input_size = 2  # Dimension of the edge features
    SRNN.human_node_output_size = 256  # Dimension of the node output

    # Embedding size
    SRNN.human_node_embedding_size = 64  # Embedding size of node features
    SRNN.human_human_edge_embedding_size = 64  # Embedding size of edge features

    # Attention vector dimension
    SRNN.attention_size = 64  # Attention size

    # training config
    training = BaseConfig()
    training.lr = 4e-5  # learning rate (default: 7e-4)
    training.eps = 1e-5  # RMSprop optimizer epsilon
    training.alpha = 0.99  # RMSprop optimizer alpha
    training.max_grad_norm = 0.5  # max norm of gradients
    training.num_env_steps = 10e6  # number of environment steps to train: 10e6 for holonomic, 20e6 for unicycle
    training.use_linear_lr_decay = False  # use a linear schedule on the learning rate: True for unicycle, False for holonomic
    training.save_interval = 200  # save interval, one save per n updates
    training.log_interval = 20  # log interval, one log per n updates
    training.use_proper_time_limits = (
        False  # compute returns taking into account time limits
    )
    training.cuda_deterministic = (
        False  # sets flags for determinism when using CUDA (potentially slow!)
    )
    training.cuda = True  # use CUDA for training
    training.num_processes = 12  # how many training CPU processes to use
    training.output_dir = "data/dummy"  # the saving directory for train.py
    training.resume = False  # resume training from an existing checkpoint or not
    training.load_path = None  # if resume = True, load from the following checkpoint
    training.overwrite = True  # whether to overwrite the output directory in training
    training.num_threads = 1  # number of threads used for intraop parallelism on CPU
