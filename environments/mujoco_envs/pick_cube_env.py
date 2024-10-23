import os
import numpy as np
import h5py

from dm_control import mjcf
from transforms3d import euler
from enum import IntEnum

from .mujoco_ur5 import UR5Robotiq, DEG2CTRL
from .mujoco_env import MujocoEnv
from ..trajectory_generator import JointTrajectory, interpolate_trajectory
from ..utils import Pose, get_best_orn_for_gripper


##### action 8, qpos 7 !!!!!


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
        random_dynamics_to_apply: list = [],
    ):
        self.random_dynamics_to_apply = random_dynamics_to_apply
        # naming convention for each property should be "<element_type>.<element_name>|<main_property_name>.<sub_property_name>.<sub_sub_property_name>...", and the value should be + and - range
        self.robot_random_dynamics = {
            # link masses
            "link mass": {
                "body.shoulder_link|inertial.mass": (0.6, -0.6),
                "body.upper_arm_link|inertial.mass": (0.6, -0.6),
                "body.forearm_link|inertial.mass": (0.6, -0.6),
                "body.wrist_1_link|inertial.mass": (0.6, -0.6),
                "body.wrist_2_link|inertial.mass": (0.6, -0.6),
                # "body.wrist_3_link|inertial.mass": (0.6, -0.6), # commented because this makes the mass of this range link negative and stops the simulation
            },

            # joint damping
            "joint damping": {
                "joint.shoulder_pan_joint|damping": (0, 2.99),
                "joint.shoulder_lift_joint|damping": (0, 2.99),
                "joint.elbow_joint|damping": (0, 2.99),
                "joint.wrist_1_joint|damping": (0, 2.99),
                "joint.wrist_2_joint|damping": (0, 2.99),
                "joint.wrist_3_joint|damping": (0, 2.99),
            },

            # # actuator gain
            # # TODO select suitable ranges
            # "actuator gain": {

            # },

            # # link inertia
            # # TODO select suitable ranges
            # "link inertia": {

            # },

            # # joint stiffness
            # # TODO select suitable ranges
            # "joint stiffness": {

            # },

            # # gravity
            # # TODO select suitable ranges
            # "gravity": {

            # },
        }

        # TODO add random dynamics to object
        # add object mass in the same format as used in the robot_random_dynamics
        self.object_random_dynamics = {
            
        }

        # TODO select suitable range
        # add gravity in the same format as used in the robot_random_dynamics
        self.world_random_dynamics = {

        }

        self.random_dynamics_groups = { 
            "robot": self.robot_random_dynamics,
            "box": self.object_random_dynamics,
            "world": self.world_random_dynamics,
        }
    
        super().__init__(
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            time_limit=time_limit,
            step_limit=step_limit,
            is_render=is_render,
        )
        # RNG seed for the pose so that it's not affected by other random processes
        self.pose_rng = np.random.default_rng(42)
        np.random.seed(42)
        self.hand_eye_cam = self.physics.model.camera("ur5e/robotiq_2f85/d435i/rgb")
        self.top_cam = self.physics.model.camera("d435i/rgb")
        self.ur5_robotiq = UR5Robotiq(self.physics, 0, "ur5e")
        self.env_max_reward = 1

    # Function to dynamically set a nested property
    def set_nested_property(self, obj, props, value):
        for prop in props[:-1]:
            obj = getattr(obj, prop)
        setattr(obj, props[-1], value)

    # Function to dynamically access properties
    def get_nested_property(self, obj, props):
        for prop in props:
            obj = getattr(obj, prop)
        return obj

    def add_random_dynamics(self, model, model_random_dynamics, random_dynamics_to_apply):
        """_summary_

        Args:
            model (dm_control.mjcf): mjcf.from_path model (e.g. model of robot, environemnt, cube, etc.)
            model_random_dynamics ( dict ): dict of dicts containing the random dynamics with the following naming convention for each property should be "<element_type>.<element_name>|<main_property_name>.<sub_property_name>.<sub_sub_property_name>...", and the value should be + and - range. 
            
            e.g.:
            model_random_dynamics = {
                "link mass": {
                    "body.shoulder_link|inertial.mass|link mass": (600, -600),
                    "body.wrist_3_link|inertial.mass|link mass": (600, -600),
                },

                "joint damping": {
                    "joint.shoulder_pan_joint|damping": (0, 2.99),
                    "joint.shoulder_lift_joint|damping": (0, 2.99),
                },
            }

            random_dynamics_to_apply ( list ): list of strings, each one of them is a dynamics type (e.g. link mass). Only the dynamics types included in this list will be applied
        
        Return:
            model with parameters changed according to changes list
        """

        for dynamics_type in random_dynamics_to_apply:
            
            if dynamics_type in model_random_dynamics:
                for dynamics_element, _range in model_random_dynamics[dynamics_type].items():
                    desc, props = dynamics_element.split("|")
                    desc_type, desc_name = desc.split(".")
                    props = props.split(".")

                    # set new value for this property/attribute
                    default_value = self.get_nested_property(model.find(desc_type, desc_name), props)
                    random_noise = np.random.uniform(*_range)
                    new_value = default_value + random_noise
                    print(f"Setting {dynamics_element} to {new_value}")
                    self.set_nested_property(model.find(f"{desc_type}", f"{desc_name}"), props, new_value)

        return model

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
                "../assets/universal_robots_ur5e/ur5e.xml",
            ),
        )
        robot_model.worldbody.light.clear()

        # add random dynamics:
        print("Applying random dynamics to robot")
        robot_model = self.add_random_dynamics(robot_model, self.robot_random_dynamics, self.random_dynamics_to_apply)

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
                    <geom type="box" size="0.02 0.02 0.02" rgba="1 0 0 1" />
                </body>
            </worldbody>
        </mujoco>"""
        )
        world_model.worldbody.attach(box_model).add(
            "joint", type="free", damping=0.01, name="obj_joint"
        )

        # robot table
        box_model2 = mjcf.from_xml_string(
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
        obj_table = mjcf.from_xml_string(
            """<mujoco>
                <worldbody>
                    <body name="box" pos="0 0 0">
                        <geom type="box" size="0.33 0.33 0.001" rgba="0.239 0.262 0.309 1" />
                    </body>
                </worldbody>
            </mujoco>"""
        )

        # 0.149 0.373 0.314
        # 0.788 0.416 0.306
        # 0.235 0.18 0.78
        # 0.582 0.688 0.698
        # 0.043 0.179 0.155

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

        self.models = {
            "world": world_model,
            "robot": robot_model,
            "gripper": gripper,
            "top_cam": top_cam,
            "wrist_cam": wrist_cam,
            "box": box_model,
            "box2": box_model2,
            "obj_table": obj_table,
        }

        physics = mjcf.Physics.from_mjcf_model(world_model)

        return physics

    def reset(self, options=None):
        # to reset the environment with different dynamics
        self.load_models()

        info = {}
        self.step_num = 0
        self.physics.reset()

        xrr = np.random.uniform(2.0, 3.0)

        # obj position limit
        random_position = [
            self.pose_rng.uniform(-0.10, 0.10),
            self.pose_rng.uniform(0.45, 0.65),
            self.pose_rng.uniform(0.1, 0.2),
        ]

        random_quat = euler.euler2quat(
            0,
            0,
            np.pi * self.pose_rng.uniform(-1, 1),
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

        print(f"Random Init Pose: {random_pose}")

        if type(options) != type(None):
            self.physics.named.data.qpos["unnamed_model/obj_joint/"] = options[
                "generated_cube_pose"
            ]
            info["generated_cube_pose"] = options["generated_cube_pose"]
        else:
            self.physics.named.data.qpos["unnamed_model/obj_joint/"] = random_pose
            info["generated_cube_pose"] = random_pose

        print("generated_cube_pose : ", info["generated_cube_pose"][0])
        # joints = np.array([-1.57, -1.57, -1.57, -1.57, 1.57, -1.57])

        joints = np.deg2rad([-87.0, -64.0, -116.0, -63.0, 90.0, -90.0])

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
        # target_qpos = action

        qpos_action = Pose(position=action[:3], orientation=action[3:7])
        target_qpos = self.ur5_robotiq.inverse_kinematics(qpos_action)

        # self.physics.data.ctrl[:-1] = target_qpos[:-1]
        # self.physics.data.ctrl[-1] = target_qpos[-1]
        self.physics.data.ctrl[:-1] = target_qpos
        self.physics.data.ctrl[-1] = action[-1]

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

    def collect_data(self, dynamic=None, options=None):
        episode_idx = 0

        current_file_path = os.getcwd()

        # Construct the dataset path
        dataset_path = os.path.join(
            os.path.dirname(os.path.dirname(current_file_path)),  # Go back two directories
            "dataset",
            "pick_cube",
            '+'.join(dynamic),  # Assuming dynamic is a list of strings
        )

        logs_path = os.path.join(dataset_path, 'logs.txt')
        
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
        else:
            # If the directory already exists, delete directory and create a new one
            import shutil
            shutil.rmtree(dataset_path)
            os.makedirs(dataset_path)
            
        with open(logs_path, 'w'):
            pass  # This will create or clean the file
        

        
        # for i in range(100):
        while episode_idx < 10:
            _, info = self.reset()  # options[i])
            (
                joint_traj,
                actions,
                qvels,
                hand_eye_frames,
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

            camera_names = ["hand_eye_cam", "top_cam"]
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
            data_dict[f"/observations/images/hand_eye_cam"] = hand_eye_frames
            data_dict[f"/observations/images/top_cam"] = top_frames

            max_timesteps = len(joint_traj)

            with h5py.File(
                os.path.join(dataset_path, f"episode_{episode_idx-1}.hdf5"),
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
                # import pudb; pudb.set_trace()
                qpos = obs.create_dataset(
                    "qpos", (max_timesteps, 7), compression="gzip", compression_opts=9
                )
                qvel = obs.create_dataset(
                    "qvel", (max_timesteps, 7), compression="gzip", compression_opts=9
                )
                action = root.create_dataset(
                    "action", (max_timesteps, 8), compression="gzip", compression_opts=9
                )

                for name, array in data_dict.items():
                    root[name][...] = array

            # add the following to the logs.. 1. the episode name, and beneath that add the following:
            # 2. all the different robot parameters that are mentioned in any random dynamics variable and their actual value while running this episode
            # 3. the init pose in this episode
            random_dynamics_actual_values = {}

            for dynamics_group_name, dynamics_group in self.random_dynamics_groups.items():
                if dynamics_group_name in self.models:
                    for dynamics_elements in dynamics_group.values():
                        for dynamics_element in dynamics_elements.keys():
                            desc, props = dynamics_element.split("|")
                            desc_type, desc_name = desc.split(".")
                            props = props.split(".")

                            actual_value = self.get_nested_property(
                                self.models[dynamics_group_name].find(desc_type, desc_name), props
                            )
                            random_dynamics_actual_values[dynamics_element] = actual_value

            log_message = f"Episode {episode_idx-1}:\n"
            for random_dynamics_actual_value_name, random_dynamics_actual_value in random_dynamics_actual_values.items():
                log_message += f"{random_dynamics_actual_value_name}: {random_dynamics_actual_value}\n"
            # log_message += f"Random Dynamics Actual Values: {random_dynamics_actual_values}\n"
            
            
            formatted_pose = [f"{x:.3f}" for x in info['generated_cube_pose']]
            log_message += f"Init Pose: {formatted_pose}\n"

            with open(logs_path, 'a') as file:
                file.write(log_message + '\n')

            # log_message = 
            # with open(logs_path, 'a') as file:
            #     file.write(log_message + '\n')



    def collect_data_sequence(self):

        # 변수 초기화
        env_state = EnvState.APPROACH

        joint_traj = []
        actions = []
        qvels = []
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
            if env_state == 4:
                break
            grip_angle = 0

            target_pos = self.ur5_robotiq.get_end_effector_pose()
            target_pos.position[:] = self.first_target_pose.position

            # 타겟 포즈 생성
            if env_state == EnvState.APPROACH:
                target_pos.position[2] = self.first_target_pose.position[2] + 0.1
            elif env_state == EnvState.PICK:
                target_pos.position[2] = self.first_target_pose.position[2]
            elif env_state == EnvState.GRASP:
                grip_angle = 250
            elif env_state == EnvState.UP:
                target_pos.position[2] = self.first_target_pose.position[2] + 0.4
                grip_angle = 250

            # target orientation
            euler_ = euler.quat2euler(cur_qpos[-4:])
            quat = euler.euler2quat(-0.0, 0.0, euler_[axis_num])  # twist

            end_effector_pose = self.ur5_robotiq.get_end_effector_pose()
            quat = get_best_orn_for_gripper(end_effector_pose.orientation, quat)

            target_pos.orientation[:] = quat

            # 경로 생성
            if env_state != EnvState.GRASP:
                waypoints = self.make_trajectory_by_pose(target_pos, 50)

            else:
                waypoints = self.close_gripper()

            waypoints_len = len(waypoints)

            for i in range(waypoints_len):
                if i >= waypoints_len:
                    i = waypoints_len - 1

                if env_state != EnvState.GRASP:
                    action = np.append(waypoints[i], grip_angle)
                    _, _, terminated, _, _ = self.step(action)  # 50

                else:
                    tcp = self.ur5_robotiq.end_effector_pose
                    pose = tcp.position
                    quat = tcp.orientation

                    tcp_np = np.append(pose, quat)

                    action = np.append(tcp_np, grip_angle)
                    # action = np.append(self.ur5_robotiq.joint_positions, grip_angle)
                    _, _, terminated, _, _ = self.step(action)  # 5

                current_grip_angle = (
                    np.rad2deg(
                        self.physics.named.data.qpos[
                            "ur5e/robotiq_2f85/right_driver_joint"
                        ]
                    )
                    * DEG2CTRL
                )

                # # joint noise
                # current_joints = self.ur5_robotiq.joint_positions
                # joint_noise = np.random.normal(0, 1, 6)
                # noise_joint = current_joints + joint_noise/100
                # joint_traj.append(
                #     np.append(noise_joint, current_grip_angle)
                # )

                joint_traj.append(
                    np.append(self.ur5_robotiq.joint_positions, current_grip_angle)
                )
                qvels.append(np.append(self.ur5_robotiq.joint_velocities, 0))

                if env_state != EnvState.GRASP:
                    actions.append(
                        np.append(waypoints[i], grip_angle)
                    )  # grip_angle = 250
                else:
                    actions.append(
                        np.append(tcp_np, waypoints[i])
                        # np.append(self.ur5_robotiq.joint_positions, waypoints[i]) # waypoints[i] = 0~ 250
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
                # print(f"is_render : {self.is_render}")
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
        )

    def make_trajectory(self, target_pose, time=0.01):

        start = np.array(self.ur5_robotiq.joint_positions)

        target_joints = self.ur5_robotiq.inverse_kinematics(target_pose)
        end = target_joints

        trajectory = JointTrajectory(start, end, time, time / 0.002, 6)

        return trajectory

    def make_trajectory_by_pose(self, target_pose, length):

        start_ee = self.ur5_robotiq.get_end_effector_pose()

        start_pose = start_ee.position
        start_quat = start_ee.orientation

        start = np.append(start_pose, start_quat)

        end_ee = target_pose
        end_pose = end_ee.position
        end_quat = end_ee.orientation

        end = np.append(end_pose, end_quat)

        # trajectory = JointTrajectory(start, end, 0.01, length, 6)
        trajectory = interpolate_trajectory(start, end, 50)  # 이거 확인해봐야함

        # print(" sample trajectory : ", trajectory)
        return trajectory

    def close_gripper(self):
        start = np.array([0])
        end = np.array([250])
        time = 0.01

        trajectory = JointTrajectory(start, end, time, 15, 6)

        return trajectory