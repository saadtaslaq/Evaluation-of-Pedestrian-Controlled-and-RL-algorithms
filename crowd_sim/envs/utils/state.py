import numpy as np
class FullState(object):
    def __init__(self, px, py, vx, vy, radius, gx, gy, v_pref, theta):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.gx = gx
        self.gy = gy
        self.v_pref = v_pref
        self.theta = theta

        self.position = (self.px, self.py)
        self.goal_position = (self.gx, self.gy)
        self.velocity = (self.vx, self.vy)

    def __add__(self, other):
        return other + (self.px,
                        self.py,
                        self.vx,
                        self.vy,
                        self.radius,
                        self.gx,
                        self.gy,
                        self.v_pref,
                        self.theta)

    def __str__(self):
        return ' '.join([str(x) for x in [self.px,
                                          self.py,
                                          self.vx,
                                          self.vy,
                                          self.radius,
                                          self.gx,
                                          self.gy,
                                          self.v_pref,
                                          self.theta]])

class FullState_sf(object):
    def __init__(self, px, py, vx, vy, gx, gy):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.gx = gx
        self.gy = gy

        self.position = (self.px, self.py)
        self.goal_position = (self.gx, self.gy)
        self.velocity = (self.vx, self.vy)

    def __add__(self, other):
        return other + (self.px,
                        self.py,
                        self.vx,
                        self.vy,
                        self.gx,
                        self.gy)

    def __str__(self):
        return ' '.join([str(x) for x in [self.px,
                                          self.py,
                                          self.vx,
                                          self.vy,
                                          self.gx,
                                          self.gy]])

    # def makearray(self):
    #     arr=[self.px,self.py,
    #                                       self.vx,
    #                                       self.vy,
    #                                       self.gx,
    #                                       self.gy]
        
    #     return arr

    # def __str__(self):
    #     return np.array(([str(x) for x in [self.px,
    #                                       self.py,
    #                                       self.vx,
    #                                       self.vy,
    #                                       self.gx,
    #             __str__                          self.gy]]))

class ObservableState(object):
    def __init__(self, px, py, vx, vy, radius):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.radius = radius

        self.position = (self.px, self.py)
        self.velocity = (self.vx, self.vy)
        # print("self.velocity", self.velocity)

    def __add__(self, other):
        return other + (self.px, self.py, self.vx, self.vy, self.radius)

    def __str__(self):
        return ' '.join(
            [str(x) for x in [self.px, self.py, self.vx, self.vy, self.radius]])


class JointState(object):
    def __init__(self, self_state, human_states):
        assert isinstance(self_state, FullState) or isinstance(self_state, FullState_sf)
        for human_state in human_states:
            assert isinstance(human_state, ObservableState)

        self.self_state = self_state
        self.human_states = human_states
