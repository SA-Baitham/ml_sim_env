from __future__ import annotations

from pydantic import dataclasses, validator
from typing import List

import numpy as np
from transforms3d import affines, euler, quaternions
from scipy.spatial.transform import Rotation

import os
from PIL import Image

LINK_SEPARATOR_TOKEN = "|"


class AllowArbitraryTypes:
    # TODO look into numpy.typing.NDArray
    # https://numpy.org/devdocs/reference/typing.html#numpy.typing.NDArray
    arbitrary_types_allowed = True


@dataclasses.dataclass(config=AllowArbitraryTypes, frozen=True)
class Pose:
    position: np.ndarray  # shape: (3, )
    orientation: np.ndarray  # shape: (4, ), quaternion

    def __hash__(self) -> int:
        return hash((*self.position.tolist(), *self.orientation.tolist()))

    @validator("position")
    @classmethod
    def position_shape(cls, v: np.ndarray):
        if v.shape != (3,):
            raise ValueError("position must be 3D")
        return v

    @validator("orientation")
    @classmethod
    def orientation_shape(cls, v: np.ndarray):
        if v.shape != (4,):
            raise ValueError("orientation must be a 4D quaternion")
        return v

    @property
    def flattened(self) -> List[float]:
        return list(self.position) + list(self.orientation)

    def __eq__(self, other) -> bool:
        return bool(
            np.allclose(self.position, other.position)
            and np.allclose(self.orientation, other.orientation)
        )

    @property
    def matrix(self) -> np.ndarray:
        return affines.compose(
            T=self.position, R=quaternions.quat2mat(self.orientation), Z=np.ones(3)
        )

    @staticmethod
    def from_matrix(matrix: np.ndarray) -> Pose:
        T, R = affines.decompose(matrix)[:2]
        return Pose(position=T.copy(), orientation=quaternions.mat2quat(R.copy()))

    def transform(self, transform_matrix: np.ndarray) -> Pose:
        assert transform_matrix.shape == (
            4,
            4,
        ), f"expected 4x4 transformation matrix but got {transform_matrix.shape}"
        T, R, _, _ = affines.decompose(transform_matrix @ self.matrix)
        return Pose(position=T, orientation=quaternions.mat2quat(R))

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        pos_str = ",".join(f"{x:.05f}" for x in self.position)
        rot_str = ",".join(f"{x:.05f}" for x in euler.quat2euler(self.orientation))
        return f"Pose(pos=({pos_str}),rot=({rot_str}))"

    @staticmethod
    def orientation_distance(q1: np.ndarray, q2: np.ndarray) -> float:
        diff = (
            Rotation.from_quat(q1[[1, 2, 3, 0]])
            * Rotation.from_quat(q2[[1, 2, 3, 0]]).inv()
        )
        dist = diff.magnitude()
        assert type(dist) == float
        return dist

    def distance(self, other: Pose, orientation_factor: float = 0.05) -> float:
        position_distance = float(np.linalg.norm(self.position - other.position))
        orientation_distance = Pose.orientation_distance(
            self.orientation, other.orientation
        )
        dist = position_distance + orientation_factor * orientation_distance
        return dist


def get_part_path(model, body) -> str:
    rootid = body.rootid
    path = ""
    while True:
        path = body.name + path
        currid = body.id
        if currid == rootid:
            return path
        body = model.body(body.parentid)
        path = LINK_SEPARATOR_TOKEN + path


def get_best_orn_for_gripper(reference_orn: np.ndarray, query_orn: np.ndarray):
    # rotate gripper about z-axis, choose the closer one
    other_orn = quaternions.qmult(
        euler.euler2quat(0, 0, np.pi),
        query_orn,
    )
    if Pose.orientation_distance(reference_orn, other_orn) < Pose.orientation_distance(
        reference_orn, query_orn
    ):
        return other_orn
    return query_orn

def frames_to_gif(dataset_path, render_frames, episode_idx, failure_idx=0):
    print("Saving gif file...")
    # save the render_frames into 
    gif_path = os.path.join(dataset_path, f"episode_{episode_idx}_{failure_idx}.gif")
    images = render_frames
    # resize images to half their resolution
    pil_images = [Image.fromarray(image) for image in images]

    # Create the GIF with a specific duration (e.g., 0.1 seconds per frame)
    pil_images[0].save(
    gif_path,
    save_all=True,
    append_images=pil_images[1:],
    duration=50,  # duration in milliseconds per frame
    loop=0  # loop indefinitely
    )