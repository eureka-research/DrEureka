from .sensor import Sensor
from isaacgym import gymapi
from isaacgym.torch_utils import *
import torch
import numpy as np

class AttachedCameraSensor(Sensor):
    def __init__(self, env, attached_robot_asset=None):
        super().__init__(env)
        self.env = env
        self.attached_robot_asset = attached_robot_asset

        

    def initialize(self, camera_label, camera_pose, camera_rpy, env_ids=None):
        if env_ids is None: env_ids = range(self.env.num_envs)

        camera_props = gymapi.CameraProperties()
        camera_props.width = self.env.cfg.perception.image_width
        camera_props.height = self.env.cfg.perception.image_height
        camera_props.horizontal_fov = self.env.cfg.perception.image_horizontal_fov


        self.cams = []

        for env_id in env_ids:

            cam = self.env.gym.create_camera_sensor(self.env.envs[env_id], camera_props)
            # initialize camera position
            # attach the camera to the base
            trans_pos = gymapi.Vec3(camera_pose[0], camera_pose[1], camera_pose[2])
            quat_pitch = quat_from_angle_axis(torch.Tensor([-camera_rpy[1]]), torch.Tensor([0, 1, 0]))[0]
            quat_yaw = quat_from_angle_axis(torch.Tensor([camera_rpy[2]]), torch.Tensor([0, 0, 1]))[0]
            quat = quat_mul(quat_yaw, quat_pitch)
            trans_quat = gymapi.Quat(quat[0], quat[1], quat[2], quat[3])
            transform = gymapi.Transform(trans_pos, trans_quat)
            follow_mode = gymapi.CameraFollowMode.FOLLOW_TRANSFORM
            self.env.gym.attach_camera_to_body(cam, self.env.envs[env_id], 0, transform, follow_mode)

            self.cams.append(cam)

        return self.cams

    def get_observation(self, env_ids = None):

        raise NotImplementedError
    
    def get_depth_images(self, env_ids = None):
        if env_ids is None: env_ids = range(self.env.num_envs)

        depth_images = []
        for env_id in env_ids:
            img = self.env.gym.get_camera_image(self.env.sim, self.env.envs[env_id], self.cams[env_id],
                                            gymapi.IMAGE_DEPTH)
            w, h = img.shape
            depth_images.append(torch.from_numpy(img.reshape([1, w, h])).to(self.env.device))
        depth_images = torch.cat(depth_images, dim=0)
        return depth_images
    
    def get_rgb_images(self, env_ids):
        if env_ids is None: env_ids = range(self.env.num_envs)

        rgb_images = []
        for env_id in env_ids:
            img = self.env.gym.get_camera_image(self.env.sim, self.env.envs[env_id], self.cams[env_id],
                                            gymapi.IMAGE_COLOR)
            w, h = img.shape
            rgb_images.append(
                torch.from_numpy(img.reshape([1, w, h // 4, 4]).astype(np.int32)).to(self.env.device))
        rgb_images = torch.cat(rgb_images, dim=0)
        return rgb_images
    
    def get_segmentation_images(self, env_ids):
        if env_ids is None: env_ids = range(self.env.num_envs)

        segmentation_images = []
        for env_id in env_ids:
            img = self.env.gym.get_camera_image(self.env.sim, self.env.envs[env_id], self.cams[env_id],
                                            gymapi.IMAGE_SEGMENTATION)
            w, h = img.shape
            segmentation_images.append(
                torch.from_numpy(img.reshape([1, w, h]).astype(np.int32)).to(self.env.device))
        segmentation_images = torch.cat(segmentation_images, dim=0)
        return segmentation_images