class Terrain:
    def __init__(self, env):
        self.env = env

    def initialize(self):
        raise NotImplementedError

    def get_friction(self, x, y):
        """ Returns the friction at a given point in the terrain.
        # """
        return self.friction_samples[x, y]
    
    def get_roughness(self, x, y):
        """ Returns the roughness at a given point in the terrain.
        # """
        return self.roughness_samples[x, y]
    
    def get_restitution(self, x, y):
        """ Returns the restitution at a given point in the terrain.
        # """
        return self.restitution_samples[x, y]
    
    def get_height(self, x, y):
        """ Returns the height at a given point in the terrain.
        # """
        return self.height_samples[x, y]
    
    