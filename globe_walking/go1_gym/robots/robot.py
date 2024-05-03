

class Robot:
    def __init__(self, env):
        self.env = env

    def apply_torques(self, torques):
        pass

    def get_torques(self):
        pass

    def get_state(self):
        pass

    def add_camera(self, camera):
        pass

    def get_num_bodies(self):
        return self.num_bodies

    def get_num_dof(self):
        return self.num_dof
    
    def get_num_actuated_dof(self):
        return self.num_actuated_dof