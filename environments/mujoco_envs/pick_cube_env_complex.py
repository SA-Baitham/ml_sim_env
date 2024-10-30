import os
import numpy as np
import h5py

from dm_control import mjcf
from transforms3d import euler
from transforms3d import quaternions
from enum import IntEnum

from .mujoco_ur5 import UR5Robotiq, DEG2CTRL
from .mujoco_env import MujocoEnv
from ..trajectory_generator import JointTrajectory
from ..utils import Pose, get_best_orn_for_gripper

from .mujoco_robot import MujocoRobot
from PIL import Image


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
        randomizations_to_apply: dict = {},
        configs: dict = {},
    ):
        self.configs = configs
        self.randomizations_to_apply = randomizations_to_apply
        self.random_dynamics = configs["random_dynamics"]

        # TODO add random dynamics to object (if needed)
        self.object_random_dynamics = {}

        # TODO add random dynamics to world (if needed)
        self.world_random_dynamics = {}

        self.random_dynamics_groups = { 
            "robot": self.random_dynamics,
            "box": self.object_random_dynamics,
            "world": self.world_random_dynamics,
        }

        self.seed = seed
        self.control_dt = control_dt
        self.physics_dt = physics_dt
        self.time_limit = time_limit
        self.step_limit = step_limit
        self.is_render = is_render

        # RNG seed for the pose so that it's not affected by other random processes
        self.pose_rng = np.random.default_rng(42)
        np.random.seed(42)

    def super_init(self, seed, control_dt, physics_dt, time_limit, step_limit, is_render):
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
        print("Loading models...")
        self.current_file_path = os.path.dirname(os.path.realpath(__file__))

        # call the robot
        robot_model = mjcf.from_path(
            os.path.join(
                self.current_file_path,
                "../assets/universal_robots_ur5e/ur5e_complex.xml",
            ),
        )
        robot_model.worldbody.light.clear()

        # add random dynamics:
        robot_model = self.add_random_dynamics(robot_model, self.random_dynamics, self.randomizations_to_apply)

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

        top_cam_pos = np.array([-1, -2, 3])
        robot_pos = robot_model.find("body", "base").pos + np.array([0, 0, 0.2])

        cam_to_robot = robot_pos - top_cam_pos

        rotation_matrix = self.rotation_matrix_to_align_z_and_x(cam_to_robot, [0, 0, 1]) # rotate z axis to cam_to_robot

        # call the top camera
        top_cam = mjcf.from_path(
            os.path.join(
                self.current_file_path,
                "../assets/realsense_d435i/d435i_with_cam.xml",
            ),
        )
        # TODO

        # call the default world
        world_model = mjcf.from_path(
            os.path.join(self.current_file_path, "../assets/default_world.xml")
        )

        # add light to the world
        if "light_source" in self.randomizations_to_apply:
            num_light_sources = np.random.randint(1, self.configs["num_light_sources"])
            for light_i in range(num_light_sources):
                print("Adding light source")
                # random light position
                light_pos_x = np.random.uniform(-10, 10)
                light_pos_y = np.random.uniform(-10, 10)
                light_pos_z = np.random.uniform(5, 20)
                light_pos = np.array([light_pos_x, light_pos_y, light_pos_z])

                print(f"Light source {light_i} position: {light_pos}")
                
                # light direction (pointing to the center of the world)
                light_pos_normalized = -light_pos / np.linalg.norm(light_pos)

                random_diffuse = np.random.uniform(0.0, 1, 3)
                random_specular = np.random.uniform(0.0, 1, 3)

                world_model.worldbody.add(
                    "light",
                    name=f"light{light_i}",
                    pos=f"{light_pos_x} {light_pos_y} {light_pos_z}",
                    dir=f"{light_pos_normalized[0]} {light_pos_normalized[1]} {light_pos_normalized[2]}",
                    diffuse=f"{random_diffuse[0]} {random_diffuse[1]} {random_diffuse[2]}",
                    specular=f"{random_specular[0]} {random_specular[1]} {random_specular[2]}",
                    castshadow="true",
                )

        spawn_site = world_model.worldbody.add(
            "site",
            pos=(top_cam_pos[0], top_cam_pos[1], top_cam_pos[2]),
            quat=quaternions.mat2quat(rotation_matrix),
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
        ################## cube color ############################
        if "object_color" in self.randomizations_to_apply:
            color = np.random.randint(0, 255, 3)
            adj_color = [round(x / 255, 3) for x in color]
            rgba_color = np.append(adj_color, 1)
        else:
            rgba_color = "1.0 0.0 0.0 1"

        box_model = mjcf.from_xml_string(
            f"""<mujoco>
            <worldbody>
                <body name="box" pos="0 0 0" >
                    <geom type="box" size="0.015 0.015 0.015" rgba="{rgba_color}" />
                </body>
            </worldbody>
        </mujoco>"""
        )
        world_model.worldbody.attach(box_model).add(
            "joint", type="free", damping=0.01, name="obj_joint"
        )

        # table under the robot
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

        if "table_color" in self.randomizations_to_apply:
            color = np.random.randint(0, 255, 3)
            adj_color = [round(x / 255, 3) for x in color]
            rgba_color = np.append(adj_color, 1)
        else:
            rgba_color = "0.239 0.262 0.309 1"

        # make a obj table
        obj_table = mjcf.from_xml_string(
            f"""<mujoco>
                <worldbody>
                    <body name="box" pos="0 0 0">
                        <geom type="box" size="0.33 0.33 0.001" rgba="{rgba_color}" />
                    </body>
                </worldbody>
            </mujoco>"""
        )

        spawn_pos = (0, 0.55, 0.0)
        spawn_site = world_model.worldbody.add("site", pos=spawn_pos, group=3)
        spawn_site.attach(obj_table)

        # Add real floor
        if "real_floor" in self.randomizations_to_apply:
            current_file_path = os.getcwd()
            floor_texture_path = os.path.join(current_file_path, "environments/assets/source/floor2.png")
            floor_pattern = mjcf.from_xml_string(
                f"""<mujoco>
                    <asset>
                        <texture name="floor_texture" type="2d" file="{floor_texture_path}" gridsize="4 3" />
                        <material name="box_material" texture="floor_texture" />
                    </asset>
                    <worldbody>
                        <body name="box" pos="0 0 0">
                            <geom type="box" size="0.54 0.72 0.0001" material="box_material" />
                        </body>
                    </worldbody>
                </mujoco>

                """
            )
            spawn_pos = (-0.05, 0.55, 0.0)
            spawn_site = world_model.worldbody.add("site", pos=spawn_pos, group=3)
            spawn_site.attach(floor_pattern)

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
            # "box2": box_model2,
            "obj_table": obj_table,
        }

        physics = mjcf.Physics.from_mjcf_model(world_model)

        return physics

    def reset(self, options=None):
        # to reset the environment with different dynamics
        self.super_init(
            seed=self.seed,
            control_dt=self.control_dt,
            physics_dt=self.physics_dt,
            time_limit=self.time_limit,
            step_limit=self.step_limit,
            is_render=self.is_render,
        )
        # self.physics = self.load_models()

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
            height=480, width=640, depth=False, camera_id=self.top_cam.id
        )
        images["hand_eye_cam"] = self.physics.render(
            height=480,
            width=640,
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

    def collect_data(self, options=None):
        episode_idx = 0

        current_file_path = os.getcwd()

        # Construct the dataset path
        dataset_path = os.path.join(
            current_file_path,
            "dataset_random_dynamics_complex",
            "pick_cube",
            ' + '.join(self.randomizations_to_apply),  # Assuming dynamic is a list/dict of strings
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

            actions_tcp = [np.append(self.ur5_robotiq.forward_kinematics(a[:-1], True).flattened, a[-1]) for a in actions]

            data_dict["/observations/qpos"] = joint_traj
            data_dict["/observations/qvel"] = qvels
            data_dict["/observations/tcp"] = ee_poses
            data_dict["/action"] = actions
            data_dict["/action_tcp"] = actions_tcp
            data_dict[f"/observations/images/hand_eye_cam"] = hand_eye_frames
            data_dict[f"/observations/images/top_cam"] = top_frames

            max_timesteps = len(joint_traj)
            # dataset_path = os.path.join(
            #     self.current_file_path,
            #     f"../../dataset/pick_cube/",
            # )
            # if not os.path.exists(dataset_path):
            #     os.makedirs(dataset_path)

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
                        (max_timesteps, 480, 640, 3),
                        dtype="uint8",
                        chunks=(1, 480, 640, 3),
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
                action_tcp = root.create_dataset(
                    "action_tcp", (max_timesteps, 8), compression="gzip", compression_opts=9
                )

                for name, array in data_dict.items():
                    root[name][...] = array

            # add random dynamics actual values to the log file
            random_dynamics_actual_values = {}

            for dynamics_group_name, dynamics_group in self.random_dynamics_groups.items():
                if dynamics_group_name in self.models:
                    for dynamics_elements in dynamics_group.values():
                        for dynamics_element in dynamics_elements.keys():
                            if dynamics_element == "gravity":
                                actual_value = self.models[dynamics_group_name].option.gravity[2]
                            else:
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
            
            
            formatted_pose = [f"{x:.3f}" for x in info['generated_cube_pose']]
            log_message += f"Init Pose: {formatted_pose}\n"

            # print robot parameters using self.physics.model

            with open(logs_path, 'a') as file:
                file.write(log_message + '\n')

            if self.configs["save_gifs"]:
                print("Saving gif file...")
                # save the top_frames into 
                gif_path = os.path.join(dataset_path, f"episode_{episode_idx-1}.gif")
                images = top_frames
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


            # log_message = 
            # with open(logs_path, 'a') as file:
            #     file.write(log_message + '\n')
            print(f"Episode {episode_idx} is saved")
            
            # close the environment
            self.close()

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
                        height=480,
                        width=640,
                        depth=False,
                        camera_id=self.hand_eye_cam.id,
                    )
                )

                top_frames.append(
                    self.physics.render(
                        height=480,
                        width=640,
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

    """
    .########.....###....##....##.########...#######..##.....##...
    .##.....##...##.##...###...##.##.....##.##.....##.###...###...
    .##.....##..##...##..####..##.##.....##.##.....##.####.####...
    .########..##.....##.##.##.##.##.....##.##.....##.##.###.##...
    .##...##...#########.##..####.##.....##.##.....##.##.....##...
    .##....##..##.....##.##...###.##.....##.##.....##.##.....##...
    .##.....##.##.....##.##....##.########...#######..##.....##...
    .............########..##....##.##....##....###....##.....##.####..######...######.
    .............##.....##..##..##..###...##...##.##...###...###..##..##....##.##....##
    .............##.....##...####...####..##..##...##..####.####..##..##.......##......
    .............##.....##....##....##.##.##.##.....##.##.###.##..##..##........######.
    .............##.....##....##....##..####.#########.##.....##..##..##.............##
    .............##.....##....##....##...###.##.....##.##.....##..##..##....##.##....##
    .............########.....##....##....##.##.....##.##.....##.####..######...######.
    """

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
        """
        Applies random dynamics to the given model based on the provided configuration.

        Args:
            model (dm_control.mjcf): The MJCF model (e.g., robot, environment, cube, etc.).
            model_random_dynamics (dict): A dictionary specifying the random dynamics to apply. 
                The dictionary should follow this structure:
                {
                    "dynamics_type": {
                        "element_type.element_name|main_property_name.sub_property_name": [
                            [lower_range, upper_range],  # Range for random noise
                            "operation"  # Operation to apply: "add", "mul", or "set"
                        ],
                        ...
                    },
                    ...
                }
                Example:
                {
                    "link mass": {
                        "body.shoulder_link|inertial.mass": [[600, -600], "add"],
                        "body.wrist_3_link|inertial.mass": [[600, -600], "add"]
                    },
                    "joint damping": {
                        "joint.shoulder_pan_joint|damping": [[0, 2.99], "add"],
                        "joint.shoulder_lift_joint|damping": [[0, 2.99], "add"]
                    }
                }

            random_dynamics_to_apply (list): A list of dynamics types to apply (e.g., ["link mass", "joint damping"]).

        Returns:
            model (dm_control.mjcf): The model with updated parameters based on the applied random dynamics.
        """

        for dynamics_type in random_dynamics_to_apply:
            
            if dynamics_type in model_random_dynamics:
                for dynamics_element, info in model_random_dynamics[dynamics_type].items():
                    if dynamics_type == "gravity":
                        default_value = model.option.gravity[2]
                    else:
                        desc, props = dynamics_element.split("|")
                        desc_type, desc_name = desc.split(".")
                        props = props.split(".")
                        default_value = self.get_nested_property(model.find(desc_type, desc_name), props)

                    # set new value for this property/attribute
                    _range = tuple(info[0])
                    operation = info[1]
                    random_noise = np.random.uniform(*_range)
                    if operation == "add":
                        new_value = default_value + random_noise
                    elif operation == "mul":
                        new_value = default_value * random_noise
                    elif operation == "set":
                        new_value = random_noise
                    # print(f"Setting {dynamics_element} to {new_value}")

                    if dynamics_type == "gravity":
                        model.option.gravity[2] = new_value
                    else:
                        self.set_nested_property(model.find(f"{desc_type}", f"{desc_name}"), props, new_value)

        return model
    
    def rotation_matrix_to_align_z_and_x(self, v_prime, plane_normal):
        """
        Calculate the rotation matrix R that aligns the z-axis with v_prime and the x-axis parallel to the plane.
        
        Parameters:
        v_prime (numpy array): Target direction for the z-axis (3D)
        plane_normal (numpy array): Normal vector of the plane (3D)
        
        Returns:
        R (numpy array): Rotation matrix (3x3)
        """
        
        # Normalize the vectors
        v_prime = v_prime / np.linalg.norm(v_prime)
        plane_normal = plane_normal / np.linalg.norm(plane_normal)
        
        # Ensure v_prime is not collinear with plane_normal
        if np.allclose(v_prime, plane_normal) or np.allclose(v_prime, -plane_normal):
            raise ValueError("v_prime cannot be collinear with the plane normal.")
        
        # Calculate the x-axis (orthogonal to both v_prime and plane_normal)
        x_axis = np.cross(plane_normal, v_prime)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        # Calculate the y-axis (orthogonal to both x_axis and v_prime)
        y_axis = np.cross(v_prime, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        # Construct the rotation matrix
        R = np.array([x_axis, y_axis, v_prime]).T
        
        return R