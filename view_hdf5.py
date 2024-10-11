import h5py
from glob import glob

# Define the path to the directory containing the HDF5 files
filename = "/home/plaif_train/syzy/motion/mo_plaif_act/dataset/pick_cube"

def folder_hdf5():
    # Find all HDF5 files in the directory
    files = glob(filename + "/*.hdf5")

    # Iterate over each file
    for file in files:
        with h5py.File(file, 'r+') as root:

            actions = root['action'] 
            observations = root['observations']
            print("  <<<    actions    >>> ")
            for action in actions:
                print(action) 
            
            print("  <<<    observations    >>> ")
            for qpos in observations['qpos']:
                print(qpos)

    
def single_hdf5():
    file = filename + "/episode_0_damping4.hdf5"
    
    with h5py.File(file, 'r+') as root:

        actions = root['action'] 
        observations = root['observations']
        print("  <<<    actions    >>> ")
        for action in actions:
            print(action.tolist()) 
        
        print("  <<<    observations    >>> ")
        for qpos in observations['qpos']:
            print(qpos.tolist())
    

if __name__ == "__main__":
    # folder_hdf5()
    single_hdf5()