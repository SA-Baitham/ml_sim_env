from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from .utils import Pose

class RobotBase(ABC):
    def __init__(self, init_joints: np.ndarray):
        self.last_joints = init_joints

    @property
    @abstractmethod
    def joint_positions(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def joint_velocities(self) -> Optional[np.ndarray]:
        pass

    @property
    @abstractmethod
    def joint_torques(self) -> Optional[np.ndarray]:
        pass

    @property
    @abstractmethod
    def end_effector_pose(self) -> Pose:
        pass

    @abstractmethod
    def move_to_joints(self, target_joints: np.ndarray): # TODO make controller
        pass

    @abstractmethod
    def forward_kinematics(
        self, joints: np.ndarray, return_ee_pose: bool = False
    ) -> Optional[Pose]:
        pass

    @abstractmethod
    def inverse_kinematics(self, pose: Pose) -> Optional[np.ndarray]:
        pass