class Asset:
    """
    Asset class
    Wraps reset, domain randomization, actuation, and observation for a
    non-robot asset
    """
    def __init__(self, env):
        self.env = env
        self.num_dof = 0  # default
        self.num_bodies = 1  # default
        self.num_actuated_dof = 0  # default

    def initialize_asset(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def randomize(self):
        raise NotImplementedError

    def get_force_feedback(self):
        raise NotImplementedError

    def get_observation(self):
        raise NotImplementedError

    def get_num_bodies(self):
        return self.num_bodies

    def get_num_dof(self):
        return self.num_dof

    def get_num_actuated_dof(self):
        return self.num_actuated_dof
