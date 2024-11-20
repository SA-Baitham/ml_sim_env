import os
import numpy as np
import h5py

from dm_control import mjcf
from dm_control.mujoco.engine import Camera
from transforms3d import euler
from transforms3d import quaternions
from enum import IntEnum

from .mujoco_ur5 import UR5Robotiq, DEG2CTRL
from .mujoco_env import MujocoEnv
from ..trajectory_generator import JointTrajectory, interpolate_trajectory
from ..utils import Pose, get_best_orn_for_gripper, frames_to_gif

from .mujoco_robot import MujocoRobot
from PIL import Image
import cv2
from scipy.spatial.transform import Rotation as R

import json

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
        
        if configs:
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

        self.random_init_poses = []

        # RNG seed for the pose so that it's not affected by other random processes
        self.pose_rng = np.random.default_rng(42)
        self.gt_pose_rng = np.random.default_rng(42)
        np.random.seed(42)

        # to visualize the trajectory path
        self.trajectory_cam = self.configs["trajectory_cam"]
        self.camera_params= []

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
        if self.configs: self.render_cam = self.physics.model.camera("d435i_render/rgb")
        self.ur5_robotiq = UR5Robotiq(self.physics, 0, "ur5e")
        self.env_max_reward = 1

        self.cam_name2id = {
            "hand_eye_cam": self.hand_eye_cam.id, 
            "top_cam": self.top_cam.id, 
            "render_cam": self.render_cam.id
            }
        

        self.cam_res = {
            "hand_eye_cam": (480 // 2, 640 // 2),
            "top_cam": (480 // 2, 640 // 2),
            "render_cam": (480, 640),
        }

        self.cameras = {cam_name: Camera(physics=self.physics, height=self.cam_res[cam_name][0], width=self.cam_res[cam_name][1], camera_id=cam_id) for cam_name, cam_id in self.cam_name2id.items()}
        

    def load_models(self):
        self.current_file_path = os.path.dirname(os.path.realpath(__file__))

        # call the robot
        robot_model = mjcf.from_path(
            os.path.join(
                self.current_file_path,
                "../assets/universal_robots_ur5e/ur5e.xml",
            ),
        )
        robot_model.worldbody.light.clear()

        # add random dynamics:
        if self.configs:
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

        # call the top camera
        top_cam = mjcf.from_path(
            os.path.join(
                self.current_file_path,
                "../assets/realsense_d435i/d435i_with_cam.xml",
            ),
        )
        # TODO

        # call the render camera
        if self.configs:
            render_cam = mjcf.from_path(
                os.path.join(
                    self.current_file_path,
                    "../assets/realsense_d435i/d435i_render.xml",
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
                # random light position
                light_pos_x = np.random.uniform(-10, 10)
                light_pos_y = np.random.uniform(-10, 10)
                light_pos_z = np.random.uniform(5, 20)
                light_pos = np.array([light_pos_x, light_pos_y, light_pos_z])
                
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

        # TODO
        # spawn the top camera
        spawn_site = world_model.worldbody.add(
            "site",
            name="top_cam",
            pos=(0.04, 0.53, 1.1),
            quat=euler.euler2quat(np.pi, 0, -np.pi / 2),
            group=3,
        )
        spawn_site.attach(top_cam)

        # spawn the render camera
        robot_model_spawn_pos = (0, 0, 0.145)
        robot_pos = np.array([robot_model_spawn_pos[0], robot_model_spawn_pos[1], robot_model_spawn_pos[2] + 0.6])
        table_model_spawn_pos = (0, 0, 0)
        table_pos = np.array([table_model_spawn_pos[0], table_model_spawn_pos[1], table_model_spawn_pos[2] + 0.35])
        
        if self.configs:
            render_cam_pos = self.configs["render_cam_pos"]
            cam_to_robot = table_pos - render_cam_pos

            rotation_matrix = self.rotation_matrix_to_align_z_and_x(cam_to_robot, [0, 0, 1]) # rotate z axis to cam_to_robot

            # TODO
            spawn_site = world_model.worldbody.add(
                "site",
                name="render_cam",
                pos=tuple(render_cam_pos),
                quat=quaternions.mat2quat(rotation_matrix),
                group=3,
            )
            spawn_site.attach(render_cam)

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
            rgba_color = np.array([1.0, 0.0, 0.0, 1])

        box_model = mjcf.from_xml_string(
            f"""<mujoco>
            <worldbody>
                <body name="box" pos="0 0 0" >
                    <geom type="box" size="0.015 0.015 0.015" rgba="{rgba_color[0]} {rgba_color[1]} {rgba_color[2]} {rgba_color[3]}" />
                </body>
            </worldbody>
        </mujoco>"""
        )
        world_model.worldbody.attach(box_model).add(
            "joint", type="free", damping=0.01, name="red_cube_joint"
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
            rgba_color = np.array([0.239, 0.262, 0.309, 1])

        # make a obj table
        obj_table = mjcf.from_xml_string(
            f"""<mujoco>
                <worldbody>
                    <body name="box" pos="{table_pos}">
                        <geom type="box" size="0.33 0.33 0.001" rgba="{rgba_color[0]} {rgba_color[1]} {rgba_color[2]} {rgba_color[3]}" />
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
        spawn_site = world_model.worldbody.add("site", pos=robot_model_spawn_pos, group=3)
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

    def reset(self, options=None, random_pose=None):
        # Initialize action positions list
        self.action_positions_used = []
        self.action_quats_used = []
        
        self.action_positions_saved = []
        self.action_quats_saved = []

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
        self.load_models()

        info = {}
        self.step_num = 0
        self.physics.reset()

        # obj position limit
        if random_pose is None:
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

        self.random_init_poses.append(random_pose)

        print(f"Random Init Pose (cm): {[int(100*val) for val in random_pose]}")

        if type(options) != type(None):
            self.physics.named.data.qpos["unnamed_model/red_cube_joint/"] = options[
                "generated_cube_pose"
            ]
            info["generated_cube_pose"] = options["generated_cube_pose"]
        else:
            self.physics.named.data.qpos["unnamed_model/red_cube_joint/"] = random_pose
            info["generated_cube_pose"] = random_pose

        joints = np.deg2rad([-87.0, -64.0, -116.0, -63.0, 90.0, -90.0])

        grip_angle = 0

        for _ in range(2000):
            self.physics.forward()
            self.physics.data.ctrl[:-1] = joints
            self.physics.data.ctrl[-1] = grip_angle
            self.physics.step()

        self.first_target_pose = Pose(
            position=np.array(
                self.physics.named.data.qpos["unnamed_model/red_cube_joint/"][:3]
            ),
            orientation=np.array(
                self.physics.named.data.qpos["unnamed_model/red_cube_joint/"][3:]
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
        if self.configs:
            images["render_cam"] = self.physics.render(
                height=480, width=640, depth=False, camera_id=self.render_cam.id
            )
        images["hand_eye_cam"] = self.physics.render(
            height=480 // 2,
            width=640 // 2,
            depth=False,
            camera_id=self.hand_eye_cam.id,
        )

        obs["qpos"] = current_joints
        obs["images"] = images

        if self.physics.named.data.qpos["unnamed_model/red_cube_joint/"][2] > 0.1:
            reward = 1
        else:
            reward = 0

        obs["reward"] = reward

        return obs

    def compute_reward(self):
        cube_z_position = self.physics.named.data.qpos["unnamed_model/red_cube_joint/"][2]

        if cube_z_position < 0.1:
            return 0
        else:
            return 1

    def step(self, action):
        # Append the current action's position
        self.action_positions_used.append(action[:3])
        self.action_quats_used.append(action[3:-1])

        qpos_action = Pose(position=action[:3], orientation=action[3:7]) # TCP space (x, y, z, quat)
        target_qpos = self.ur5_robotiq.inverse_kinematics(qpos_action) # joint space (joint1, joint2, ..., joint6)

        self.physics.data.ctrl[:-1] = target_qpos # self.physics.data.ctrl has joint space + gripper, 7 values
        self.physics.data.ctrl[-1] = action[-1]

        for _ in range(40):  # why does it need to be 40???? so we have only 40 steps to reach the target? that means that if we increased that number then whatever random dynamics 
            self.physics.step()
            self.physics.forward()

        obs = self.get_observations()
        reward = self.compute_reward()
        terminated = self.step_limit_exceeded() or self.time_limit_exceeded()
        truncated = False  # TODO:Make Joint Limit
        info = {}

        self.step_num += 1

        return obs, reward, terminated, truncated, info
    
    def generate_gt_from_real_failure(self, options=None):
        episode_idx = 0
        failure_idx = 0

        current_file_path = os.getcwd()

        # Construct the dataset path
        dataset_path = os.path.join(
            current_file_path,
            self.configs["gt_dataset_dir_name"],
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

        # read ood_eval.json
        with open(os.path.join(current_file_path, "ood_eval.json"), "r") as f:
            gt_init_poses = json.load(f)

        gt_init_poses = gt_init_poses["episodes"]
        failure_case_indices = set(self.configs["failure_case_indices"])

        episode_idx = 0

        for failure_episode_idx, gt_init_pose in enumerate(gt_init_poses):
            if failure_episode_idx not in failure_case_indices:
                continue
            for episode_version in range(self.configs["num_gt_noise_poses"]):
                
                print(f"EPISODE: {failure_episode_idx}_{episode_version}")

                init_pose = np.array(gt_init_pose)

                # perturbation array:
                if episode_version != 0:
                    x_perturbation = self.gt_pose_rng.normal(0, self.configs["x_perturbation_range"])
                    y_perturbation = self.gt_pose_rng.normal(0, self.configs["y_perturbation_range"])
                    perturbation = np.array([x_perturbation, y_perturbation, 0, 0, 0, 0, 0]) # TODO add perturbation to quaternion values as well

                    # TODO: check if we need to correct the quaternion values coming from the json file since they are real quaternion values
                    init_pose = init_pose + perturbation
                    init_pose = self.perturb_quat(init_pose)

                _, info = self.reset(random_pose=init_pose)  # options[i])
                (
                    joint_traj,
                    actions,
                    qvels,
                    hand_eye_frames,
                    top_frames,
                    hand_eye_depth_frames,
                    top_depth_frames,
                    render_frames,
                    env_state,
                ) = self.collect_data_sequence()
                if (
                    self.physics.named.data.qpos["unnamed_model/red_cube_joint/"][2] < 0.1
                    or env_state < 3
                ):
                    print("FAILED")
                    failure_idx += 1
                    if self.configs["save_gifs"] and self.configs["save_failed_gifs"]:
                        frames_to_gif(dataset_path, render_frames, episode_idx+1, failure_idx)
                        
                    continue
                else:
                    print("SUCCEED")
                    episode_idx += 1
                    failure_idx = 0
                    
                if self.configs["save_gifs"]:
                    frames_to_gif(dataset_path, render_frames, failure_episode_idx, episode_version)
                # 성공 trajectory 생성
                # ================================================================================================ #
                # 데이터 구조 설정

                camera_names = ["hand_eye_cam", "top_cam"]
                if self.configs["depth"]:
                    camera_names += ["hand_eye_depth_cam", "top_depth_cam"]
                    
                data_dict = {
                    "/observations/qpos": [],
                    "/observations/qvel": [],
                    "/action": [],
                }
                for cam_name in camera_names:
                    data_dict[f"/observations/images/{cam_name}"] = []

                if "salt_and_pepper" in self.randomizations_to_apply:
                    hand_eye_frames = self.add_salt_and_pepper_noise(hand_eye_frames)
                    top_frames = self.add_salt_and_pepper_noise(top_frames)
                
                if "HSV" in self.randomizations_to_apply:
                    hand_eye_frames = self.randomize_hsv(hand_eye_frames)
                    top_frames = self.randomize_hsv(top_frames)
                
                if "normalize" in self.randomizations_to_apply:
                    hand_eye_frames = self.normalize_images(hand_eye_frames)
                    top_frames = self.normalize_images(top_frames)

                data_dict["/observations/qpos"] = joint_traj
                data_dict["/observations/qvel"] = qvels
                data_dict["/action"] = actions
                data_dict[f"/observations/images/hand_eye_cam"] = hand_eye_frames
                data_dict[f"/observations/images/top_cam"] = top_frames
                if self.configs["depth"]:
                    data_dict[f"/observations/images/hand_eye_depth_cam"] = hand_eye_depth_frames
                    data_dict[f"/observations/images/top_depth_cam"] = top_depth_frames

                max_timesteps = len(joint_traj)

                with h5py.File(
                    os.path.join(dataset_path, f"episode_{failure_episode_idx}_{episode_version}.hdf5"),
                    "w",
                    rdcc_nbytes=1024**2 * 2,
                ) as root:
                    root.attrs["sim"] = True
                    root.attrs["info"] = info["generated_cube_pose"]
                    obs = root.create_group("observations")
                    image = obs.create_group("images")
                    for cam_name in camera_names:
                        if 'depth' in cam_name:
                            _ = image.create_dataset(
                            cam_name, (max_timesteps, 480 // 2, 640 // 2), compression="gzip", compression_opts=9
                            )
                        else:
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
                        "action", (max_timesteps, 8), compression="gzip", compression_opts=9
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

                # log_message = f"Episode {episode_idx-1}:\n"
                # for random_dynamics_actual_value_name, random_dynamics_actual_value in random_dynamics_actual_values.items():
                #     log_message += f"{random_dynamics_actual_value_name}: {random_dynamics_actual_value}\n"            
                
                formatted_pose = [f"{x:.3f}" for x in info['generated_cube_pose']]
                # log_message += f"Init Pose: {formatted_pose}\n"
                log_message = f"{formatted_pose}\n"

                with open(logs_path, 'a') as file:
                    file.write(log_message)

                print(f"Episode {failure_episode_idx}_{episode_version} is saved")

                # close the environment
                self.close()

    def collect_data(self, options=None):
        episode_idx = 0
        failure_idx = 0

        current_file_path = os.getcwd()

        # Construct the dataset path
        dataset_path = os.path.join(
            current_file_path,
            self.configs["dataset_dir_name"],
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

        while episode_idx < self.configs["num_episodes"]:
            print(f"EPISODE: {episode_idx}")
            _, info = self.reset()  # options[i])
            (
                joint_traj,
                actions,
                qvels,
                hand_eye_frames,
                top_frames,
                hand_eye_depth_frames,
                top_depth_frames,
                render_frames,
                env_state,
            ) = self.collect_data_sequence()
            if (
                self.physics.named.data.qpos["unnamed_model/red_cube_joint/"][2] < 0.1
                or env_state < 3
            ):
                print("FAILED")
                failure_idx += 1
                if self.configs["save_gifs"] and self.configs["save_failed_gifs"]:
                    frames_to_gif(dataset_path, render_frames, episode_idx+1, failure_idx)
                    
                continue
            else:
                print("SUCCEED")
                episode_idx += 1
                failure_idx = 0
                
            if self.configs["save_gifs"]:
                frames_to_gif(dataset_path, render_frames, episode_idx, failure_idx)
            # 성공 trajectory 생성
            # ================================================================================================ #
            # 데이터 구조 설정

            camera_names = ["hand_eye_cam", "top_cam"]
            if self.configs["depth"]:
                camera_names += ["hand_eye_depth_cam", "top_depth_cam"]
                
            data_dict = {
                "/observations/qpos": [],
                "/observations/qvel": [],
                "/action": [],
            }
            for cam_name in camera_names:
                data_dict[f"/observations/images/{cam_name}"] = []

            if "salt_and_pepper" in self.randomizations_to_apply:
                hand_eye_frames = self.add_salt_and_pepper_noise(hand_eye_frames)
                top_frames = self.add_salt_and_pepper_noise(top_frames)
            
            if "HSV" in self.randomizations_to_apply:
                hand_eye_frames = self.randomize_hsv(hand_eye_frames)
                top_frames = self.randomize_hsv(top_frames)
            
            if "normalize" in self.randomizations_to_apply:
                hand_eye_frames = self.normalize_images(hand_eye_frames)
                top_frames = self.normalize_images(top_frames)

            data_dict["/observations/qpos"] = joint_traj
            data_dict["/observations/qvel"] = qvels
            data_dict["/action"] = actions
            data_dict[f"/observations/images/hand_eye_cam"] = hand_eye_frames
            data_dict[f"/observations/images/top_cam"] = top_frames
            if self.configs["depth"]:
                data_dict[f"/observations/images/hand_eye_depth_cam"] = hand_eye_depth_frames
                data_dict[f"/observations/images/top_depth_cam"] = top_depth_frames

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
                    if 'depth' in cam_name:
                        _ = image.create_dataset(
                        cam_name, (max_timesteps, 480 // 2, 640 // 2), compression="gzip", compression_opts=9
                        )
                    else:
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
                    "action", (max_timesteps, 8), compression="gzip", compression_opts=9
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

            # log_message = f"Episode {episode_idx-1}:\n"
            # for random_dynamics_actual_value_name, random_dynamics_actual_value in random_dynamics_actual_values.items():
            #     log_message += f"{random_dynamics_actual_value_name}: {random_dynamics_actual_value}\n"            
            
            formatted_pose = [f"{x:.3f}" for x in info['generated_cube_pose']]
            # log_message += f"Init Pose: {formatted_pose}\n"
            log_message = f"{formatted_pose}\n"

            with open(logs_path, 'a') as file:
                file.write(log_message)

            print(f"Episode {episode_idx} is saved")

            # close the environment
            self.close()

    def collect_data_sequence(self):

        # 변수 초기화
        env_state = EnvState.APPROACH

        joint_traj = []
        actions = []
        qvels = []
        hand_eye_frames = []
        top_frames = []
        render_frames = []
        hand_eye_depth_frames = []
        top_depth_frames = []

        # 초기상태 - 현재 관절 위치
        cur_qpos = self.physics.named.data.qpos["unnamed_model/red_cube_joint/"]
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
                
                # What would make i >= waypoints_len???? why this line is here???
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

                joint_traj.append(
                    np.append(self.ur5_robotiq.joint_positions, current_grip_angle)
                )
                qvels.append(np.append(self.ur5_robotiq.joint_velocities, 0))

                # Save actions in a way that matches the real robot
                if env_state != EnvState.GRASP:
                    
                    # Converting quaternion from real robot to mujoco criteria
                    if self.configs["make_real_same_as_mujoco"]:
                        corrected_quat = self.correct_real_quaternion(waypoints[i][3:7])
                        waypoints[i][3:7] = corrected_quat
                    elif self.configs["make_mujoco_same_as_real"]:
                        corrected_quat = self.correct_mujoco_quaternion(waypoints[i][3:7])
                        waypoints[i][3:7] = corrected_quat
                    actions.append(
                        np.append(waypoints[i], grip_angle)
                    )  # grip_angle = 250
                    

                else:
                    tcp = self.ur5_robotiq.end_effector_pose
                    pose = tcp.position
                    quat = tcp.orientation

                    # Converting quaternion from real robot to mujoco criteria
                    if self.configs["make_real_same_as_mujoco"]:
                        corrected_quat = self.correct_real_quaternion(quat)
                        quat = corrected_quat
                    elif self.configs["make_mujoco_same_as_real"]:
                        corrected_quat = self.correct_mujoco_quaternion(quat)
                        quat = corrected_quat

                    tcp_np = np.append(pose, quat)
                    actions.append(
                        np.append(tcp_np, waypoints[i])
                        # np.append(self.ur5_robotiq.joint_positions, waypoints[i]) # waypoints[i] = 0~ 250
                    )


                # Append the current action's position
                self.action_positions_saved.append(actions[-1][:3])
                self.action_quats_saved.append(actions[-1][3:-1])

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

                if self.configs["depth"]:
                    hand_eye_depth = self.physics.render(
                        height=480 // 2,
                        width=640 // 2,
                        depth=True,
                        camera_id=self.hand_eye_cam.id,
                    )

                    top_depth = self.physics.render(
                        height=480 // 2,
                        width=640 // 2,
                        depth=True,
                        camera_id=self.top_cam.id,
                    )

                    hand_eye_depth_frames.append(hand_eye_depth)
                    top_depth_frames.append(top_depth)



                if self.configs:
                    render_frames.append(
                        self.physics.render(
                            height=480,
                            width=640,
                            depth=False,
                            camera_id=self.render_cam.id,
                        )
                    )


                # Visualize trajectory path
                if self.trajectory_cam:
                    camera_params = {}
                    camera_params['cam_intrinsic'] = self.physics.model.cam_intrinsic[self.cam_name2id[self.trajectory_cam]]
                    camera_params['cam_pos'] = self.physics.model.site(self.trajectory_cam).pos
                    camera_params['cam_quat'] = self.physics.model.site(self.trajectory_cam).quat
                    self.camera_params.append(camera_params)

                if self.trajectory_cam == "render_cam":
                    if self.configs["show_used_ee_axis"]:
                        render_frames[-1] = self.visualize_trajectory_path(self.action_positions_used, self.action_quats_used, render_frames[-1], self.cameras[self.trajectory_cam].matrices())
                    if self.configs["show_saved_ee_axis"]:
                        render_frames[-1] = self.visualize_trajectory_path(self.action_positions_saved, self.action_quats_saved, render_frames[-1], self.cameras[self.trajectory_cam].matrices(), lightness=0.5)
                elif self.trajectory_cam == "top_cam":
                    if self.configs["show_used_ee_axis"]:
                        top_frames[-1] = self.visualize_trajectory_path(self.action_positions_used, self.action_quats_used, top_frames[-1], self.cameras[self.trajectory_cam].matrices())
                    if self.configs["show_saved_ee_axis"]:
                        top_frames[-1] = self.visualize_trajectory_path(self.action_positions_saved, self.action_quats_saved, top_frames[-1], self.cameras[self.trajectory_cam].matrices(), lightness=0.5)

                # Display the frames
                if self.configs["render"]:
                    self.render()
            env_state += 1

        return (
            joint_traj,
            actions,
            qvels,
            hand_eye_frames,
            top_frames,
            hand_eye_depth_frames,
            top_depth_frames,
            render_frames,
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

        return trajectory

    def close_gripper(self):
        start = np.array([0])
        end = np.array([250])
        time = 0.01

        trajectory = JointTrajectory(start, end, time, 15, 6)

        return trajectory
    
    #pick_cube_env.py
    def correct_mujoco_quaternion(self, mujoco_endeff_quat):
        """
        Correct the quaternion from MuJoCo to match the real robot criteria.
        Args:
            mujoco_endeff_quat (np.array): The quaternion from MuJoCo.
        Returns:
            corrected_quat (np.array): The corrected quaternion.
        """

        world_quat = np.array([0, 0, 1, 0])
        local_quat = np.array([1, 0, 0, 0])
        correction_rot = R.from_quat(local_quat) * R.from_quat(world_quat).inv()
        # Apply the correction rotation to the quaternion from MuJoCo
        corrected_quat = correction_rot * R.from_quat(mujoco_endeff_quat)
        return corrected_quat.as_quat()
    
    def correct_real_quaternion(self, real_endeff_quat):
        """
        Correct the quaternion from the real robot to match the MuJoCo criteria.
        Args:
            real_endeff_quat (np.array): The quaternion from the real robot.
        Returns:
            corrected_quat (np.array): The corrected quaternion.
        """

        world_quat = np.array([0, 0, 1, 0])
        local_quat = np.array([1, 0, 0, 0])
        correction_rot = R.from_quat(local_quat) * R.from_quat(world_quat).inv()
        # Apply the correction rotation to the quaternion from the real robot
        corrected_quat = correction_rot.inv() * R.from_quat(real_endeff_quat)
        return corrected_quat.as_quat()

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

                    if dynamics_type == "gravity":
                        model.option.gravity[2] = new_value
                    else:
                        self.set_nested_property(model.find(f"{desc_type}", f"{desc_name}"), props, new_value)

        return model
    
    """
    .##.....##.####..######..####..#######..##....##
    .##.....##..##..##....##..##..##.....##.###...##
    .##.....##..##..##........##..##.....##.####..##
    .##.....##..##...######...##..##.....##.##.##.##
    ..##...##...##........##..##..##.....##.##..####
    ...##.##....##..##....##..##..##.....##.##...###
    ....###....####..######..####..#######..##....##
    .............########.....###....##....##.########...#######..##.....##.####.########....###....########.####..#######..##....##
    .............##.....##...##.##...###...##.##.....##.##.....##.###...###..##.......##....##.##......##.....##..##.....##.###...##
    .............##.....##..##...##..####..##.##.....##.##.....##.####.####..##......##....##...##.....##.....##..##.....##.####..##
    .............########..##.....##.##.##.##.##.....##.##.....##.##.###.##..##.....##....##.....##....##.....##..##.....##.##.##.##
    .............##...##...#########.##..####.##.....##.##.....##.##.....##..##....##.....#########....##.....##..##.....##.##..####
    .............##....##..##.....##.##...###.##.....##.##.....##.##.....##..##...##......##.....##....##.....##..##.....##.##...###
    .............##.....##.##.....##.##....##.########...#######..##.....##.####.########.##.....##....##....####..#######..##....##
    """

    def normalize_images(self, image_list):
        normalized_images = []
        for image in image_list:
            # Convert image to float type for division
            image = image.astype(np.float32)
            # Normalize to range 0-1
            normalized_image = image / 255.0
            normalized_images.append(normalized_image)
        return normalized_images
    # Usage
    
    def add_salt_and_pepper_noise(self, image_list, salt_vs_pepper=0.5, amount=0.04):
        noisy_images = []
        for image in image_list:
            # Generate unique noise level for each image
            image_noise_amount = np.random.uniform(0, amount)
            # Copy the image to avoid modifying the original
            noisy_image = np.copy(image)
            # Number of pixels to alter
            num_salt = int(np.ceil(image_noise_amount * image.size * salt_vs_pepper))
            num_pepper = int(np.ceil(image_noise_amount * image.size * (1.0 - salt_vs_pepper)))
            # Generate 2D coordinates for salt and pepper
            coords_salt = [np.random.randint(0, i, num_salt) for i in image.shape[:2]]
            coords_pepper = [np.random.randint(0, i, num_pepper) for i in image.shape[:2]]
            # Add salt noise (white, typically 255 for uint8 images)
            noisy_image[tuple(coords_salt) + (slice(None),)] = 255
            # Add pepper noise (black, 0 for uint8 images)
            noisy_image[tuple(coords_pepper) + (slice(None),)] = 0
            noisy_images.append(noisy_image)
        return noisy_images
    
    def randomize_hsv(self, image_list, hue_variation=0.1, saturation_variation=0.3, value_variation=0.3):
        randomized_images = []
        for image in image_list:
            # Convert to HSV color space
            hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            # Randomly adjust each HSV channel
            h_variation = np.random.uniform(-hue_variation, hue_variation)
            s_variation = np.random.uniform(-saturation_variation, saturation_variation)
            v_variation = np.random.uniform(-value_variation, value_variation)
            # Apply variations
            hsv_image = hsv_image.astype(np.float32)
            hsv_image[..., 0] = (hsv_image[..., 0] + h_variation * 180) % 180  # Hue adjustment
            hsv_image[..., 1] = np.clip(hsv_image[..., 1] + s_variation * 255, 0, 255)  # Saturation adjustment
            hsv_image[..., 2] = np.clip(hsv_image[..., 2] + v_variation * 255, 0, 255)  # Value adjustment
            # Convert back to RGB color space
            randomized_image = cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2RGB)
            randomized_images.append(randomized_image)
        return randomized_images
    


    """
    .########..########.##....##.########..########.########..####.##....##..######..
    .##.....##.##.......###...##.##.....##.##.......##.....##..##..###...##.##....##.
    .##.....##.##.......####..##.##.....##.##.......##.....##..##..####..##.##.......
    .########..######...##.##.##.##.....##.######...########...##..##.##.##.##...####
    .##...##...##.......##..####.##.....##.##.......##...##....##..##..####.##....##.
    .##....##..##.......##...###.##.....##.##.......##....##...##..##...###.##....##.
    .##.....##.########.##....##.########..########.##.....##.####.##....##..######..
    .............########.##.....##.##....##..######..########.####..#######..##....##..######.
    .............##.......##.....##.###...##.##....##....##.....##..##.....##.###...##.##....##
    .............##.......##.....##.####..##.##..........##.....##..##.....##.####..##.##......
    .............######...##.....##.##.##.##.##..........##.....##..##.....##.##.##.##..######.
    .............##.......##.....##.##..####.##..........##.....##..##.....##.##..####.......##
    .............##.......##.....##.##...###.##....##....##.....##..##.....##.##...###.##....##
    .............##........#######..##....##..######.....##....####..#######..##....##..######.
    """
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

    def map_position_to_image(self, trajectory_3D_path, cam_matrices):
        """
        Map 3D positions to 2D image coordinates using CameraMatrices.

        Parameters:
        - trajectory_3D_path (list or ndarray): 3D positions (N x 3)
        - cam_matrices (CameraMatrices): The camera matrices (image, focal, rotation, translation)

        Returns:
        - image_coords (numpy array): 2D image coordinates (N x 2)
        """

        trajectory_3D_path = np.array(trajectory_3D_path)
        num_points = trajectory_3D_path.shape[0]

        # Build the camera projection matrix P
        P = cam_matrices.image @ cam_matrices.focal @ cam_matrices.rotation @ cam_matrices.translation

        # Convert 3D points to homogeneous coordinates
        trajectory_3D_path_h = np.hstack((trajectory_3D_path, np.ones((num_points, 1))))  # Shape (N x 4)

        # Project the 3D points
        image_coords_h = (P @ trajectory_3D_path_h.T).T  # Shape (N x 3)

        # Normalize by the third coordinate to get pixel coordinates
        image_coords = image_coords_h[:, :2] / image_coords_h[:, [2]]

        return image_coords


    
    def visualize_trajectory_path(self, trajectory_3D_path, trajecotry_quats, image, cam_matrices, lightness=0):
        """
        Visualize the trajectory path on the image with a color gradient.
        
        Parameters:
        - trajectory_3D_path (numpy array): 3D positions (N x 3)
        - image (numpy array): Image (H x W x 3)
        - cam_matrices (CameraMatrices): The camera matrices
        
        Returns:
        - image (numpy array): Image with the trajectory path
        """

        # Copy the image to avoid modifying the original
        image = np.copy(image)
        
        # Map the 3D positions to 2D image coordinates
        image_coords = self.map_position_to_image(trajectory_3D_path, cam_matrices)
        
        # Ensure that the coordinates are within image boundaries
        h, w = image.shape[:2]
        image_coords = np.clip(image_coords, [0, 0], [w - 1, h - 1])
        
        num_points = len(image_coords)
        
        # Define start and end colors (BGR format)
        start_color = np.array([255, 0, 255])      # Black (past points)
        end_color = np.array([255, 0, 0])      # Bright red (recent points)
        
        # Draw the trajectory path with a color gradient
        for i in range(num_points - 1):
            # Calculate the ratio (0 to 1) of the current point
            ratio = i / (num_points - 1)
            
            # Interpolate between the start and end colors
            color = (1 - ratio) * start_color + ratio * end_color
            color = color.astype(int).tolist()
            
            # Draw the line segment
            pt1 = tuple(image_coords[i].astype(int))
            pt2 = tuple(image_coords[i + 1].astype(int))
            cv2.line(image, pt1, pt2, color, 1, cv2.LINE_AA)

        # Add end-effector frame at the end of the trajectory
        ee_quat = trajecotry_quats[-1]
        ee_pos = trajectory_3D_path[-1]

        # original x, y, z axis
        x_axis = np.array([0.05*(1-lightness), 0, 0])
        y_axis = np.array([0, 0.05*(1-lightness), 0])
        z_axis = np.array([0, 0, 0.05*(1-lightness)])

        # Rotate the axes based on the end-effector quaternion
        rot = R.from_quat(ee_quat)
        x_axis = rot.apply(x_axis)
        y_axis = rot.apply(y_axis)
        z_axis = rot.apply(z_axis)

        # Draw the end-effector frame
        ee_origin = self.map_position_to_image([ee_pos], cam_matrices)[0]
        ee_origin = tuple(ee_origin.astype(int))
        x_axis_end = self.map_position_to_image([ee_pos + x_axis], cam_matrices)[0]
        x_axis_end = tuple(x_axis_end.astype(int))
        y_axis_end = self.map_position_to_image([ee_pos + y_axis], cam_matrices)[0]
        y_axis_end = tuple(y_axis_end.astype(int))
        z_axis_end = self.map_position_to_image([ee_pos + z_axis], cam_matrices)[0]
        z_axis_end = tuple(z_axis_end.astype(int))

        cv2.line(image, ee_origin, x_axis_end, (255*lightness, 255*lightness, 255), 2)  # Red: x-axis
        cv2.line(image, ee_origin, y_axis_end, (255*lightness, 255, 255*lightness), 2)  # Green: y-axis
        cv2.line(image, ee_origin, z_axis_end, (255, 255*lightness, 255*lightness), 2)  # Blue: z-axis

        # Draw global x, y, z axis
        global_x_axis = np.array([0.15, 0, 0])
        global_y_axis = np.array([0, 0.15, 0])
        global_z_axis = np.array([0, 0, 0.15])

        global_x_axis_end = self.map_position_to_image([global_x_axis], cam_matrices)[0]
        global_x_axis_end = tuple(global_x_axis_end.astype(int))
        global_y_axis_end = self.map_position_to_image([global_y_axis], cam_matrices)[0]
        global_y_axis_end = tuple(global_y_axis_end.astype(int))
        global_z_axis_end = self.map_position_to_image([global_z_axis], cam_matrices)[0]
        global_z_axis_end = tuple(global_z_axis_end.astype(int))

        global_origin = self.map_position_to_image([[0, 0, 0]], cam_matrices)[0]
        global_origin = tuple(global_origin.astype(int))

        cv2.line(image, global_origin, global_x_axis_end, (255*lightness, 255*lightness, 255), 2)  # Red: x-axis
        cv2.line(image, global_origin, global_y_axis_end, (255*lightness, 255, 255*lightness), 2)
        cv2.line(image, global_origin, global_z_axis_end, (255, 255*lightness, 255*lightness), 2)
        
        return image
    

    """
    ..######...########
    .##....##.....##...
    .##...........##...
    .##...####....##...
    .##....##.....##...
    .##....##.....##...
    ..######......##...
    ...........######...########.##....##.########.########.....###....########.####..#######..##....##
    ..........##....##..##.......###...##.##.......##.....##...##.##......##.....##..##.....##.###...##
    ..........##........##.......####..##.##.......##.....##..##...##.....##.....##..##.....##.####..##
    ..........##...####.######...##.##.##.######...########..##.....##....##.....##..##.....##.##.##.##
    ..........##....##..##.......##..####.##.......##...##...#########....##.....##..##.....##.##..####
    ..........##....##..##.......##...###.##.......##....##..##.....##....##.....##..##.....##.##...###
    ...........######...########.##....##.########.##.....##.##.....##....##....####..#######..##....##
"""

    def perturb_quat(self, pose):
        """
        Perturb a quaternion with Gaussian noise.
        
        Parameters:
        - quat (numpy array): Pose
        - noise_level (float): Standard deviation of the Gaussian noise
        
        Returns:
        - perturbed_quat (numpy array): Perturbed quaternion (4D)
        """

        quat = pose[-4:]
        
        # Perturb the quaternion with Gaussian noise
        noise = self.gt_pose_rng.normal(0, self.configs["quat_perturbation_range"])

        # convert quat to euler
        euler_angles = np.array(euler.quat2euler(quat))

        # perturb yaw only
        euler_angles[2] += noise

        # convert euler back to quat
        perturbed_quat = euler.euler2quat(*euler_angles)

        pose[-4:] = perturbed_quat
        
        return pose