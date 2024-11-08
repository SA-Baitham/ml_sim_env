#!/bin/bash

# datasets_options:
#     {'clean'}, # Done
#     # Dynamic parameters
#     {"link_mass"}, # Done
#     {"joint_damping"}, # Done
#     {"joint_friction"}, # Done
#     {"actuator_gain"}, # Done
#     {"link_inertia"}, # Done
#     {"joint_stiffness"}, # Done
#     {'gravity'}, # Done
    
#     # Vision parameters
#     {"light_source"}, # Done
#     {"object_color"}, # Done
#     {"robot_color"}, # Done
#     {"table_color"}, # Done
#     {"real_floor"},
#     

#     # Not added yet
#     {"HSV"},
#     {"salt_and_pepper"},
#     {"SAM"},

# List of dataset paths
dataset_paths=(
    ######## Ahmed
    # "clean"
    # "link_mass"
    # "joint_damping"
    # "joint_friction"
    # "actuator_gain"
    # "link_inertia"


    ######### Surjeet
    # "joint_stiffness"
    # "gravity"
    # "light_source"
    # "object_color"
    # "robot_color"
    # "table_color"


    ######### Server
    "real_floor"
    "HSV"
    "salt_and_pepper"
    "normalize"



    # "All"
    # "All_dynamics"
    # "All_vision"
)

# Loop through each dataset path
for dataset_path in "${dataset_paths[@]}"; do
    echo "Training with dataset: $dataset_path"
    python train_act.py "task_config.dataset_dir=/home/ahmed/ml_sim_env/dataset_for_training_corrected_orientation/pick_cube/$dataset_path"
done