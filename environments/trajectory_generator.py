from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

import numpy as np 


def CubicTimeScaling(Tf, t):
    """Computes s(t) for a cubic time scaling

    :param Tf: Total time of the motion in seconds from rest to rest
    :param t: The current time t satisfying 0 < t < Tf
    :return: The path parameter s(t) corresponding to a third-order
             polynomial motion that begins and ends at zero velocity

    Example Input:
        Tf = 2
        t = 0.6
    Output:
        0.216
    """
    return 3 * (1.0 * t / Tf) ** 2 - 2 * (1.0 * t / Tf) ** 3

def QuinticTimeScaling(Tf, t):
    """Computes s(t) for a quintic time scaling

    :param Tf: Total time of the motion in seconds from rest to rest
    :param t: The current time t satisfying 0 < t < Tf
    :return: The path parameter s(t) corresponding to a fifth-order
             polynomial motion that begins and ends at zero velocity and zero
             acceleration

    Example Input:
        Tf = 2
        t = 0.6
    Output:
        0.16308
    """
    return 10 * (1.0 * t / Tf) ** 3 - 15 * (1.0 * t / Tf) ** 4 \
           + 6 * (1.0 * t / Tf) ** 5

def JointTrajectory(thetastart, thetaend, Tf, N, method):
    """Computes a straight-line trajectory in joint space

    :param thetastart: The initial joint variables
    :param thetaend: The final joint variables
    :param Tf: Total time of the motion in seconds from rest to rest
    :param N: The number of points N > 1 (Start and stop) in the discrete
              representation of the trajectory
    :param method: The time-scaling method, where 3 indicates cubic (third-
                   order polynomial) time scaling and 5 indicates quintic
                   (fifth-order polynomial) time scaling
    :return: A trajectory as an N x n matrix, where each row is an n-vector
             of joint variables at an instant in time. The first row is
             thetastart and the Nth row is thetaend . The elapsed time
             between each row is Tf / (N - 1)

    Example Input:
        thetastart = np.array([1, 0, 0, 1, 1, 0.2, 0, 1])
        thetaend = np.array([1.2, 0.5, 0.6, 1.1, 2, 2, 0.9, 1])
        Tf = 4
        N = 6
        method = 3
    Output:
        np.array([[     1,     0,      0,      1,     1,    0.2,      0, 1]
                  [1.0208, 0.052, 0.0624, 1.0104, 1.104, 0.3872, 0.0936, 1]
                  [1.0704, 0.176, 0.2112, 1.0352, 1.352, 0.8336, 0.3168, 1]
                  [1.1296, 0.324, 0.3888, 1.0648, 1.648, 1.3664, 0.5832, 1]
                  [1.1792, 0.448, 0.5376, 1.0896, 1.896, 1.8128, 0.8064, 1]
                  [   1.2,   0.5,    0.6,    1.1,     2,      2,    0.9, 1]])
    """
    N = int(N)
    timegap = Tf / (N - 1.0)
    traj = np.zeros((len(thetastart), N))
    for i in range(N):
        if method == 3:
            s = CubicTimeScaling(Tf, timegap * i)
        else:
            s = QuinticTimeScaling(Tf, timegap * i)
        traj[:, i] = s * np.array(thetaend) + (1 - s) * np.array(thetastart)
    traj = np.array(traj).T
    return traj


def interpolate_trajectory(start_pose, target_pose, num_steps=50):
    """
    Generates a list of interpolated poses between start_pose and target_pose.

    Args:
    - start_pose: List containing [x, y, z, qw, qx, qy, qz]
    - target_pose: List containing [x, y, z, qw, qx, qy, qz]
    - num_steps: Number of steps/poses in the trajectory

    Returns:
    - List of interpolated poses, each pose is a list [x, y, z, qw, qx, qy, qz]
    """
    trajectory = []
    # Interpolate positions linearly
    for step in range(num_steps):
        fraction = step / (num_steps - 1)  # Fraction of interpolation
        interp_position = [
            start_pose[i] + fraction * (target_pose[i] - start_pose[i])
            for i in range(3)
        ]  # Comment from JH: So primitive.

        # Interpolate quaternions using Slerp
        times = [0, 1]
        key_rots = R.from_quat(
            [np.roll(start_pose[3:], -1), np.roll(target_pose[3:], -1)]
        )  # or scalar_first=True and remove np.roll if scipy version ^1.14
        slerp = Slerp(times, key_rots)
        interp_quat = slerp([fraction]).as_quat()[0]

        interp_pose = interp_position + np.roll(interp_quat, 1).tolist()
        trajectory.append(interp_pose)

    return trajectory