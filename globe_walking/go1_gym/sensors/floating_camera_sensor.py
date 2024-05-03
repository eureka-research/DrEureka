from .sensor import Sensor
from isaacgym import gymapi

class FloatingCameraSensor(Sensor):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

        camera_props = gymapi.CameraProperties()
        camera_props.width = self.env.cfg.env.recording_width_px
        camera_props.height = self.env.cfg.env.recording_height_px
        self.rendering_camera = self.env.gym.create_camera_sensor(self.env.envs[0], camera_props)
        self.env.gym.set_camera_location(self.rendering_camera, self.env.envs[0], gymapi.Vec3(1.5, 1, 3.0),
                                        gymapi.Vec3(0, 0, 0))

    def set_position(self, target_loc=None, cam_distance=None):
        if cam_distance is None:
            cam_distance = [0, -1.0, 1.0]
        if target_loc is None:
            bx, by, bz = self.env.root_states[0, 0], self.env.root_states[0, 1], self.env.root_states[0, 2]
            target_loc = [bx, by, bz]
        self.env.gym.set_camera_location(self.rendering_camera, self.env.envs[0], gymapi.Vec3(target_loc[0] + cam_distance[0],
                                                                                      target_loc[1] + cam_distance[1],
                                                                                      target_loc[2] + cam_distance[2]),
                                     gymapi.Vec3(target_loc[0], target_loc[1], target_loc[2]))

    def get_observation(self, env_ids = None):
        self.env.gym.step_graphics(self.env.sim)
        self.env.gym.render_all_camera_sensors(self.env.sim)
        img = self.env.gym.get_camera_image(self.env.sim, self.env.envs[0], self.rendering_camera, gymapi.IMAGE_COLOR)
        w, h = img.shape
        return img.reshape([w, h // 4, 4])