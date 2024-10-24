import os
import numpy as np
import h5py

from dm_control import mjcf
from transforms3d import euler
from enum import IntEnum

from .mujoco_ur5 import UR5Robotiq, DEG2CTRL
from .mujoco_env import MujocoEnv
from ..trajectory_generator import JointTrajectory
from ..utils import Pose, get_best_orn_for_gripper


class EnvState(IntEnum):
    APPROACH = 0
    PICK = 1
    GRASP = 2
    UP = 3


class PickCubeEnv(MujocoEnv):
    def __init__(
        self,
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = float("inf"),
        step_limit: int = 180,
        is_render: bool = False,
    ):
        super().__init__(
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            time_limit=time_limit,
            step_limit=step_limit,
            is_render=is_render,
        )
        self.hand_eye_cam = self.physics.model.camera("ur5e/robotiq_2f85/d435i/rgb")
        self.top_cam = self.physics.model.camera("d435i/rgb")
        self.ur5_robotiq = UR5Robotiq(self.physics, 0, "ur5e")
        self.env_max_reward = 1

    def load_models(self):
        self.current_file_path = os.path.dirname(os.path.realpath(__file__))
        # call the default world
        world_model = mjcf.from_path(
            os.path.join(self.current_file_path, "../assets/default_world.xml")
        )
        # call the robot
        robot_model = mjcf.from_path(
            os.path.join(
                self.current_file_path,
                "../assets/universal_robots_ur5e/ur5e_complex.xml",
            ),
        )
        robot_model.worldbody.light.clear()
        attachment_site = robot_model.find("site", "attachment_site")
        assert attachment_site is not None

        # call the gripper
        gripper = mjcf.from_path(
            os.path.join(
                self.current_file_path,
                "../assets/robotiq_2f85/2f85.xml",
            ),
        )

        attachment_site.attach(gripper)

        # gripper flange
        flange_model = mjcf.from_xml_string(  # TODO make function to create object
            """<mujoco>
            <worldbody>
                <body name="box" pos="0 0.02 0">
                    <geom type="box" size="0.035 0.04 0.005" rgba="0.06 0.06 0.06 1" />
                </body>
            </worldbody>
        </mujoco>"""
        )
        attachment_site.attach(flange_model)

        # call the top camera
        top_cam = mjcf.from_path(
            os.path.join(
                self.current_file_path,
                "../assets/realsense_d435i/d435i_with_cam.xml",
            ),
        )
        # TODO
        spawn_site = world_model.worldbody.add(
            "site",
            pos=(0.04, 0.53, 1.1),
            quat=euler.euler2quat(np.pi, 0, -np.pi / 2),
            group=3,
        )
        spawn_site.attach(top_cam)

        # hand eye camera / wrist camera

        wrist_cam_mount_site = gripper.find("site", "cam_mount")
        wrist_cam = mjcf.from_path(
            os.path.join(
                self.current_file_path,
                "../assets/realsense_d435i/d435i_with_cam.xml",
            ),
        )
        wrist_cam_mount_site.attach(wrist_cam)

        # Red cube for picking
        box_model = mjcf.from_xml_string(
            """<mujoco>
            <worldbody>
                <body name="box" pos="0 0 0" >
                    <geom type="box" size="0.015 0.015 0.015" rgba="1 0 0 1" />
                </body>
            </worldbody>
        </mujoco>"""
        )
        world_model.worldbody.attach(box_model).add(
            "joint", type="free", damping=0.01, name="obj_joint"
        )

        # # table under the robot
        # box_model2 = mjcf.from_xml_string(
        #     """<mujoco>
        #         <asset>
        #             <material name="shiny" specular="0.5" shininess="0.8" />
        #         </asset>
        #         <worldbody>
        #             <body name="box" pos="0 0 0">
        #                 <geom type="box" size="0.255 0.255 0.145" rgba="0.2 0.2 0.2 1" material="shiny" />
        #             </body>
        #         </worldbody>
        #     </mujoco>"""
        # )

        # spawn_pos = (0, 0, 0.0)
        # spawn_site = world_model.worldbody.add("site", pos=spawn_pos, group=3)
        # spawn_site.attach(box_model2)

        # make a obj table
        obj_table = mjcf.from_xml_string(
            """<mujoco>
                <worldbody>
                    <body name="box" pos="0 0 0">
                        <geom type="box" size="0.33 0.33 0.001" rgba="0.239 0.262 0.309 1" />
                    </body>
                </worldbody>
            </mujoco>"""
        )

        spawn_pos = (0, 0.55, 0.0)
        spawn_site = world_model.worldbody.add("site", pos=spawn_pos, group=3)
        spawn_site.attach(obj_table)

        # # floor pattern
        # floor_pattern = mjcf.from_xml_string(
        #     """<mujoco>
        #         <asset>
        #             <texture name="floor_texture" type="2d" file="/home/plaif_train/syzy/motion/mo_plaif_act/environments/assets/source/floor2.png" gridsize="4 3" />
        #             <material name="box_material" texture="floor_texture" />
        #         </asset>
        #         <worldbody>
        #             <body name="box" pos="0 0 0">
        #                 <geom type="box" size="0.54 0.72 0.0001" material="box_material" />
        #             </body>
        #         </worldbody>
        #     </mujoco>

        #     """
        # )
        # spawn_pos = (-0.05, 0.55, 0.0)
        # spawn_site = world_model.worldbody.add("site", pos=spawn_pos, group=3)
        # spawn_site.attach(floor_pattern)

        # Set the robot position
        spawn_pos = (0, 0, 0.145)
        spawn_site = world_model.worldbody.add("site", pos=spawn_pos, group=3)
        spawn_site.attach(robot_model)

        physics = mjcf.Physics.from_mjcf_model(world_model)

        return physics

    def reset(self, options=None):
        info = {}
        self.step_num = 0
        self.physics.reset()

        ############################################################3
        # # obj position limit
        # random_position = [
        #     np.random.uniform(-0.10, 0.10),
        #     np.random.uniform(0.45, 0.65),
        #     np.random.uniform(0.1, 0.2),
        # ]

        # random_quat = euler.euler2quat(
        #     0,
        #     0,
        #     np.pi * np.random.uniform(-1, 1),
        # )

        # random_pose = [
        #     random_position[0],
        #     random_position[1],
        #     random_position[2],
        #     random_quat[0],
        #     random_quat[1],
        #     random_quat[2],
        #     random_quat[3],
        # ]

        random_pose = [
            0.06956875051242348,
            0.5495662419585472,
            0.12040477176070047,
            0.7020425752804891,
            0.0,
            0.0,
            0.712134974912438,
        ]

        if type(options) != type(None):
            self.physics.named.data.qpos["unnamed_model/obj_joint/"] = options[
                "generated_cube_pose"
            ]
            info["generated_cube_pose"] = options["generated_cube_pose"]
        else:
            self.physics.named.data.qpos["unnamed_model/obj_joint/"] = random_pose
            info["generated_cube_pose"] = random_pose

        joints = np.deg2rad([-180, -180, 150, -210, -180, -180])

        grip_angle = 0

        for _ in range(2000):
            self.physics.forward()
            self.physics.data.ctrl[:-1] = joints
            self.physics.data.ctrl[-1] = grip_angle
            self.physics.step()

        self.first_target_pose = Pose(
            position=np.array(
                self.physics.named.data.qpos["unnamed_model/obj_joint/"][:3]
            ),
            orientation=np.array(
                self.physics.named.data.qpos["unnamed_model/obj_joint/"][3:]
            ),
        )
        self.env_state = EnvState.APPROACH
        obs = self.get_observations()

        return obs, info

    def get_observations(self):
        obs = {}
        images = {}

        current_grip_angle = (
            np.rad2deg(
                self.physics.named.data.qpos["ur5e/robotiq_2f85/right_driver_joint"]
            )
            * DEG2CTRL
        )
        current_joints = np.append(self.ur5_robotiq.joint_positions, current_grip_angle)

        images["top_cam"] = self.physics.render(
            height=480 // 2, width=640 // 2, depth=False, camera_id=self.top_cam.id
        )
        images["hand_eye_cam"] = self.physics.render(
            height=480 // 2,
            width=640 // 2,
            depth=False,
            camera_id=self.hand_eye_cam.id,
        )

        obs["qpos"] = current_joints
        obs["images"] = images

        if self.physics.named.data.qpos["unnamed_model/obj_joint/"][2] > 0.1:
            reward = 1
        else:
            reward = 0

        obs["reward"] = reward

        return obs

    def compute_reward(self):
        cube_z_position = self.physics.named.data.qpos["unnamed_model/obj_joint/"][2]

        if cube_z_position < 0.1:
            return 0
        else:
            return 1

    def step(self, action):
        target_qpos = action

        self.physics.data.ctrl[:-1] = target_qpos[:-1]
        self.physics.data.ctrl[-1] = target_qpos[-1]
        for _ in range(40):  # TODO: check!
            self.physics.step()
            self.physics.forward()

        obs = self.get_observations()
        reward = self.compute_reward()
        terminated = self.step_limit_exceeded() or self.time_limit_exceeded()
        truncated = False  # TODO:Make Joint Limit
        info = {}

        self.step_num += 1

        return obs, reward, terminated, truncated, info

    def collect_data(self):
        episode_idx = 0
        for i in range(100):
            _, info = self.reset()
            (
                joint_traj,
                actions,
                qvels,
                hand_eye_frames,
                top_frames,
                env_state,
                ee_poses,
            ) = self.collect_data_sequence()
            # if (
            #     self.physics.named.data.qpos["unnamed_model/obj_joint/"][2] < 0.1
            #     or env_state < 3
            # ):
            #     print("FAILED")
            #     continue
            # else:
            #     print("SUCCEED")
            #     episode_idx += 1
            # 성공 trajectory 생성
            # ================================================================================================ #
            # 데이터 구조 설정
            episode_idx += 1
            camera_names = ["hand_eye_cam", "top_cam"]
            data_dict = {
                "/observations/qpos": [],
                "/observations/qvel": [],
                "/observations/tcp": [],
                "/action": [],
            }
            for cam_name in camera_names:
                data_dict[f"/observations/images/{cam_name}"] = []

            data_dict["/observations/qpos"] = joint_traj
            data_dict["/observations/qvel"] = qvels
            data_dict["/observations/tcp"] = ee_poses
            data_dict["/action"] = actions
            data_dict[f"/observations/images/hand_eye_cam"] = hand_eye_frames
            data_dict[f"/observations/images/top_cam"] = top_frames

            max_timesteps = len(joint_traj)
            dataset_path = os.path.join(
                self.current_file_path,
                f"../../dataset/pick_cube/",
            )
            if not os.path.exists(dataset_path):
                os.makedirs(dataset_path)

            with h5py.File(
                dataset_path + f"episode_{episode_idx-1}.hdf5",
                "w",
                rdcc_nbytes=1024**2 * 2,
            ) as root:
                root.attrs["sim"] = True
                root.attrs["info"] = info["generated_cube_pose"]
                obs = root.create_group("observations")
                image = obs.create_group("images")
                for cam_name in camera_names:
                    _ = image.create_dataset(
                        cam_name,
                        (max_timesteps, 480 // 2, 640 // 2, 3),
                        dtype="uint8",
                        chunks=(1, 480 // 2, 640 // 2, 3),
                        compression="gzip",
                        compression_opts=9,
                    )
                qpos = obs.create_dataset(
                    "qpos", (max_timesteps, 7), compression="gzip", compression_opts=9
                )
                qvel = obs.create_dataset(
                    "qvel", (max_timesteps, 7), compression="gzip", compression_opts=9
                )
                action = root.create_dataset(
                    "action", (max_timesteps, 7), compression="gzip", compression_opts=9
                )
                tcp = obs.create_dataset(
                    "tcp", (max_timesteps, 8), compression="gzip", compression_opts=9
                )

                for name, array in data_dict.items():
                    root[name][...] = array

            print(f"Episode {episode_idx} is saved")

    def collect_data_sequence(self):

        # 변수 초기화
        env_state = EnvState.APPROACH

        joint_traj = []
        actions = []
        qvels = []
        ee_poses = []
        hand_eye_frames = []
        top_frames = []

        # 초기상태 - 현재 관절 위치
        cur_qpos = self.physics.named.data.qpos["unnamed_model/obj_joint/"]
        rot_mat = euler.quat2mat(cur_qpos[-4:])
        for i, num in enumerate(rot_mat[2, :]):
            if (num <= -0.9) or (num >= 0.9):
                axis_num = i
        euler_ = euler.quat2euler(cur_qpos[-4:])

        terminated = False

        while not terminated:
            if env_state == 1:
                break
            grip_angle = 0

            target_jnt = np.deg2rad([180, 0, -150, 30, 180, 180])
            waypoints = self.make_jnt2jnt_trajectory(target_jnt, 0.3)
            waypoints_len = len(waypoints)

            for i in range(waypoints_len):
                if i >= waypoints_len:
                    i = waypoints_len - 1

                if env_state != EnvState.GRASP:
                    action = np.append(waypoints[i], grip_angle)
                    _, _, terminated, _, _ = self.step(action)  # 50

                else:
                    action = np.append(self.ur5_robotiq.joint_positions, grip_angle)
                    _, _, terminated, _, _ = self.step(action)  # 5

                current_grip_angle = (
                    np.rad2deg(
                        self.physics.named.data.qpos[
                            "ur5e/robotiq_2f85/right_driver_joint"
                        ]
                    )
                    * DEG2CTRL
                )
                joint_traj.append(
                    np.append(self.ur5_robotiq.joint_positions, current_grip_angle)
                )
                qvels.append(np.append(self.ur5_robotiq.joint_velocities, 0))

                ee = self.ur5_robotiq.get_end_effector_pose()
                ee_pose = ee.position
                ee_quat = ee.orientation
                tcp = np.append(ee_pose, ee_quat)

                ee_poses.append(np.append(tcp, 0))

                if env_state != EnvState.GRASP:
                    actions.append(
                        np.append(waypoints[i], grip_angle)
                    )  # grip_angle = 250
                else:
                    actions.append(
                        np.append(
                            self.ur5_robotiq.joint_positions, waypoints[i]
                        )  # waypoints[i] = 0~ 250
                    )

                hand_eye_frames.append(
                    self.physics.render(
                        height=480 // 2,
                        width=640 // 2,
                        depth=False,
                        camera_id=self.hand_eye_cam.id,
                    )
                )

                top_frames.append(
                    self.physics.render(
                        height=480 // 2,
                        width=640 // 2,
                        depth=False,
                        camera_id=self.top_cam.id,
                    )
                )
                if self.is_render:
                    self.render()
            env_state += 1

        return (
            joint_traj,
            actions,
            qvels,
            hand_eye_frames,
            top_frames,
            env_state,
            ee_poses,
        )

    def make_trajectory(self, target_pose, time=0.01):

        target_joints = self.ur5_robotiq.inverse_kinematics(target_pose)

        start = np.array(self.ur5_robotiq.joint_positions)
        end = target_joints

        trajectory = JointTrajectory(start, end, time, time / 0.002, 6)

        return trajectory

    def make_jnt2jnt_trajectory(self, target_jnt, time=0.01):

        start = np.array(self.ur5_robotiq.joint_positions)
        end = target_jnt

        trajectory = JointTrajectory(start, end, time, time / 0.002, 6)

        return trajectory

    def close_gripper(self):
        start = np.array([0])
        end = np.array([250])
        time = 0.01

        trajectory = JointTrajectory(start, end, time, 15, 6)

        return trajectory
