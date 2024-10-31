#!/bin/bash

# datasets_options:
#     {'clean'}, 
#     # Dynamic parameters
#     {"link_mass"}, 
#     {"joint_damping"}, 
#     {"joint_friction"}, 
#     {"actuator_gain"},
#     {"link_inertia"},
#     {"joint_stiffness"},
#     {'gravity'},
    
#     # Vision parameters
#     {"light_source"},
#     {"object_color"},
#     {"robot_color"},
#     {"table_color"},
#     {"real_floor"},
#     

#     # Not added yet
#     {"HSV"},
#     {"salt_and_pepper"},
#     {"SAM"},

# List of dataset paths
dataset_paths=(
    "clean"
    "link_mass"
    "joint_damping"
    "joint_friction"
)

# Loop through each dataset path
for dataset_path in "${dataset_paths[@]}"; do
    echo "Training with dataset: $dataset_path"
    python train_act.py "task_config.dataset_dir=/home/ahmed/Desktop/workspace/ml_sim_env/dataset_for_training/pick_cube/$dataset_path"
done