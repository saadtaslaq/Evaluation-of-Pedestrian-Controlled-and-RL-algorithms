class Timeout(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'Timeout'


class ReachGoal(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'Reaching goal'
class ReachGoal_Ped(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'Reaching goal ped'

class Danger(object):
    def __init__(self, min_dist):
        self.min_dist = min_dist

    def __str__(self):
        return 'Too close'

class Deadlock(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'Deadlock occur'


class Collision(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'Collision'

class CollisionOtherAgent(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'Collision from other agent'


class Nothing(object):
    def __init__(self):
        pass

    def __str__(self):
        return ''

class Collision_Pedestrian(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'Pedestrian collided'

