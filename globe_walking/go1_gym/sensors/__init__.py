from .sensor import Sensor
from .attached_camera_sensor import AttachedCameraSensor
from .floating_camera_sensor import FloatingCameraSensor
from .joint_position_sensor import JointPositionSensor
from .joint_velocity_sensor import JointVelocitySensor
from .orientation_sensor import OrientationSensor
from .heightmap_sensor import HeightmapSensor
from .rc_sensor import RCSensor
from .action_sensor import ActionSensor
from .last_action_sensor import LastActionSensor
from .clock_sensor import ClockSensor
from .yaw_sensor import YawSensor
from .object_sensor import ObjectSensor
from .timing_sensor import TimingSensor
from .body_velocity_sensor import BodyVelocitySensor
from .object_velocity_sensor import ObjectVelocitySensor
from .restitution_sensor import RestitutionSensor
from .friction_sensor import FrictionSensor
from .feet_contact_sensor import FeetContactSensor
from .com_sensor import CoMSensor
from .payload_sensor import PayloadSensor
from .motor_strength_sensor import MotorStrengthSensor

ALL_SENSORS = {
    # "AttachedCameraSensor": AttachedCameraSensor,
    # "FloatingCameraSensor": FloatingCameraSensor,
    "JointPositionSensor": JointPositionSensor,
    "JointVelocitySensor": JointVelocitySensor,
    "OrientationSensor": OrientationSensor,
    # "HeightmapSensor": HeightmapSensor,
    "RCSensor": RCSensor,
    "ActionSensor": ActionSensor,
    "LastActionSensor": LastActionSensor,
    "ClockSensor": ClockSensor,
    "YawSensor": YawSensor,
    "ObjectSensor": ObjectSensor,
    "TimingSensor": TimingSensor,
    "BodyVelocitySensor": BodyVelocitySensor,
    "ObjectVelocitySensor": ObjectVelocitySensor,
    "RestitutionSensor": RestitutionSensor,
    "FrictionSensor": FrictionSensor,
    "FeetContactSensor": FeetContactSensor,
    "CoMSensor": CoMSensor,
    "PayloadSensor": PayloadSensor,
    "MotorStrengthSensor": MotorStrengthSensor,
}
