import os
import numpy as np
import h5py

from dm_control import mjcf
from transforms3d import euler
from enum import IntEnum

from .mujoco_rb5 import RB5Robotiq, DEG2CTRL
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

        # self.top_cam = self.physics.model.camera("d435i/rgb")
        self.wrist_cam = self.physics.model.camera("rb5_850e/robotiq_2f85/d435i/rgb")
        self.rb5_robotiq = RB5Robotiq(self.physics, 0, "rb5_850e")
        self.env_max_reward = 1

    def load_models(self):
        self.current_file_path = os.path.dirname(
            os.path.realpath(__file__)
        ) 
        # call the default world 
        world_model = mjcf.from_path(
            os.path.join(self.current_file_path, "../assets/default_world.xml")
        )
        # call the robot
        robot_model = mjcf.from_path(
            os.path.join(
                self.current_file_path,
                "../assets/rainbow_rb5_850e/rb5_850e.xml", 
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
        
        # call the top camera
        top_cam = mjcf.from_path(
            os.path.join(
                self.current_file_path,
                "../assets/realsense_d435i/d435i_with_cam.xml",
            ),
        )
        spawn_site = world_model.worldbody.add(
            "site", pos=(-0 ,-0.54 ,1.15), quat=euler.euler2quat(np.pi, 0, np.pi/2), group=3
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

        # make a red cube for picking
        box_model = mjcf.from_xml_string(  # TODO make function to create object
            """<mujoco>
            <worldbody>
                <body name="box" pos="0 0 0">
                    <geom type="box" size="0.03 0.03 0.03" rgba="1 0 0 1" />
                </body>
            </worldbody>
        </mujoco>"""
        )
        world_model.worldbody.attach(box_model).add(
            "joint", type="free", damping=0.01, name="obj_joint"
        )
        
        # make a table under the robot 
        box_model2 = mjcf.from_xml_string(  # TODO make function to create object
            """<mujoco>
                <asset>
                    <material name="shiny" specular="0.5" shininess="0.8" />
                </asset>
                <worldbody>
                    <body name="box" pos="0 0 0">
                        <geom type="box" size="0.255 0.255 0.145" rgba="0.2 0.2 0.2 1" material="shiny" />
                    </body>
                </worldbody>
            </mujoco>"""
        )

        spawn_pos = (0, 0, 0.0)
        spawn_site = world_model.worldbody.add("site", pos=spawn_pos, group=3)
        spawn_site.attach(box_model2)
        
        # make a obj table
        obj_table = mjcf.from_xml_string(  # TODO make function to create object
            """<mujoco>
                <worldbody>
                    <body name="box" pos="0 0 0">
                        <geom type="box" size="0.3 0.3 0.001" rgba="0.27 0.26 0.28 1" />
                    </body>
                </worldbody>
            </mujoco>"""
        )

        spawn_pos = (0, -0.55, 0.0)
        spawn_site = world_model.worldbody.add("site", pos=spawn_pos, group=3)
        spawn_site.attach(obj_table)
        
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

        # obj position limit
        random_position = [
            np.random.uniform(-0.28, 0.28),
            np.random.uniform(-0.85, -0.3),
            np.random.uniform(0.1, 0.2),
        ]

        random_quat = euler.euler2quat(
            0,
            0,
            np.pi * np.random.uniform(-1, 1),
        )

        random_pose = [
            random_position[0],
            random_position[1],
            random_position[2],
            random_quat[0],
            random_quat[1],
            random_quat[2],
            random_quat[3],
        ]

        if(type(options) != type(None)):
            self.physics.named.data.qpos["unnamed_model/obj_joint/"] = options["generated_cube_pose"]
            info["generated_cube_pose"] = options["generated_cube_pose"]
        else:
            self.physics.named.data.qpos["unnamed_model/obj_joint/"] = random_pose
            info["generated_cube_pose"] = random_pose

        # real robot init joint =[ 90, 0 ,-150, 60, -90, 90]
        joints = np.deg2rad([
                80.0,
                20.0,
                -100.0,
                -4.0,
                -90.0,
                190.0
            ]) 
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
                self.physics.named.data.qpos["rb5_850e/robotiq_2f85/right_driver_joint"]
            )
            * DEG2CTRL
        )
        current_joints = np.append(self.rb5_robotiq.joint_positions, current_grip_angle)
        # images["top_cam"] = self.physics.render(
        #     height=480, width=640, depth=False, camera_id=self.top_cam.id
        # )
        images["top_cam"] = self.physics.render(
            height=480, width=640, depth=False, camera_id=self.wrist_cam.id
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
        for _ in range(40):
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
                top_frames,
                env_state,
            ) = self.collect_data_sequence()
            if (
                self.physics.named.data.qpos["unnamed_model/obj_joint/"][2] < 0.1
                or env_state < 3
            ):
                print("FAILED")
                continue
            else:
                print("SUCCEED")
                episode_idx += 1
# 성공 trajectory 생성
# ================================================================================================ #
# 데이터 구조 설정

            camera_names = ["top_cam"]
            data_dict = {
                "/observations/qpos": [],
                "/observations/qvel": [],
                "/action": [],
            }
            for cam_name in camera_names:
                data_dict[f"/observations/images/{cam_name}"] = []

            data_dict["/observations/qpos"] = joint_traj
            data_dict["/observations/qvel"] = qvels
            data_dict["/action"] = actions
            data_dict[f"/observations/images/top_cam"] = top_frames

            max_timesteps = len(joint_traj)
            dataset_path = os.path.join(
                self.current_file_path,
                f"../../dataset/pick_cube/",
            )
            if not os.path.exists(dataset_path):
                os.makedirs(dataset_path)

            with h5py.File(
                dataset_path + f"episode_{episode_idx-1}.hdf5", "w", rdcc_nbytes=1024**2 * 2
            ) as root:
                root.attrs["sim"] = True
                root.attrs["info"] = info['generated_cube_pose']
                obs = root.create_group("observations")
                image = obs.create_group("images")
                for cam_name in camera_names:
                    _ = image.create_dataset(
                        cam_name,
                        (max_timesteps, 480, 640, 3),
                        dtype="uint8",
                        chunks=(1, 480, 640, 3),
                        compression='gzip', compression_opts=9
                    )
                qpos = obs.create_dataset("qpos", (max_timesteps, 7), compression='gzip', compression_opts=9)
                qvel = obs.create_dataset("qvel", (max_timesteps, 7), compression='gzip', compression_opts=9)
                action = root.create_dataset("action", (max_timesteps, 7), compression='gzip', compression_opts=9)

                for name, array in data_dict.items():
                    root[name][...] = array

    def collect_data_sequence(self):

        # 변수 초기화
        env_state = EnvState.APPROACH

        joint_traj = []   
        actions = []
        qvels = []
        top_frames = []

        # 초기상태 - 현재 관절 위치
        cur_qpos = self.physics.named.data.qpos["unnamed_model/obj_joint/"]
        rot_mat = euler.quat2mat(cur_qpos[-4:])
        for i, num in enumerate(rot_mat[2, :]):
            if (num <= -0.9) or (num >= 0.9):
                axis_num = i
        euler_ = euler.quat2euler(cur_qpos[-4:])

        terminated = False
        # 끝날때까지 envstate 0,1,2,3,4 0,1,2,3,4 ...
        while not terminated:
            grip_angle = 0

            target_pos = self.rb5_robotiq.get_end_effector_pose()
            target_pos.position[:] = self.first_target_pose.position

            # 타겟 포즈 생성
            if env_state == EnvState.APPROACH:
                target_pos.position[2] = self.first_target_pose.position[2] + 0.1
            elif env_state == EnvState.PICK:
                target_pos.position[2] = self.first_target_pose.position[2]
            elif env_state == EnvState.GRASP:
                grip_angle = 250
            else:
                target_pos.position[2] = self.first_target_pose.position[2] + 0.2
                grip_angle = 250

            # target orientation 
            euler_ = euler.quat2euler(cur_qpos[-4:])
            quat = euler.euler2quat(-0.0, 0.0, euler_[axis_num])

            end_effector_pose = self.rb5_robotiq.get_end_effector_pose()
            quat = get_best_orn_for_gripper(end_effector_pose.orientation, quat)

            target_pos.orientation[:] = quat

            # 경로 생성
            if env_state != EnvState.GRASP:
                waypoints = self.make_trajectory(target_pos, 0.1)

            else:
                waypoints = self.close_gripper()

            waypoints_len = len(waypoints)
            
            for i in range(waypoints_len):
                if i >= waypoints_len:
                    i = waypoints_len - 1

                if env_state != EnvState.GRASP:
                    action = np.append(waypoints[i], grip_angle)
                    _, _, terminated, _, _ = self.step(action) #50

                else:
                    action = np.append(self.rb5_robotiq.joint_positions, grip_angle)
                    _, _, terminated, _, _ = self.step(action) #5
                # ??  이 값은 뭘까
                current_grip_angle = (
                    np.rad2deg(
                        self.physics.named.data.qpos[
                            "rb5_850e/robotiq_2f85/right_driver_joint"
                        ]
                    )
                    * DEG2CTRL
                )
                joint_traj.append(
                    np.append(self.rb5_robotiq.joint_positions, current_grip_angle)
                )
                qvels.append(np.append(self.rb5_robotiq.joint_velocities, 0))

                if env_state != EnvState.GRASP:
                    actions.append(np.append(waypoints[i], grip_angle)) # grip_angle = 250
                else:
                    actions.append(
                        np.append(self.rb5_robotiq.joint_positions, waypoints[i]) # waypoints[i] = 0~ 250
                    )

                top_frames.append(
                    self.physics.render(
                        height=480, width=640, depth=False, camera_id=self.wrist_cam.id
                    )
                )
                if self.is_render:
                    self.render()
            env_state += 1

        return (
            joint_traj,
            actions,
            qvels,
            top_frames,
            env_state,
        )

    def make_trajectory(self, target_pose, time=0.01):
        target_joints = self.rb5_robotiq.inverse_kinematics(target_pose)

        start = np.array(self.rb5_robotiq.joint_positions)
        end = target_joints

        trajectory = JointTrajectory(start, end, time, time / 0.002, 6)

        return trajectory

    def close_gripper(self):
        start = np.array([0])
        end = np.array([250])
        time = 0.01

        trajectory = JointTrajectory(start, end, time, time / 0.002, 6)

        return trajectory