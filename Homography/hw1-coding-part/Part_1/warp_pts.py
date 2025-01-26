import numpy as np
from est_homography import est_homography

def warp_pts(X, Y, interior_pts):
    """
    First compute homography from video_pts to logo_pts using X and Y,
    and then use this homography to warp all points inside the soccer goal

    Input:
        X: 4x2 matrix of (x,y) coordinates of goal corners in video frame
        Y: 4x2 matrix of (x,y) coordinates of logo corners in penn logo
        interior_pts: Nx2 matrix of points inside goal
    Returns:
        warped_pts: Nx2 matrix containing new coordinates for interior_pts.
        These coordinate describe where a point inside the goal will be warped
        to inside the penn logo. For this assignment, you can keep these new
        coordinates as float numbers.

    """

    # You should Complete est_homography first!
    
    H = est_homography(X, Y)

    # x_min, x_max = np.min(X[:, 0]), np.max(X[:, 0])
    # y_min, y_max = np.min(X[:, 1]), np.max(X[:, 1])

    # # Generate a grid of points between (x_min, y_min) and (x_max, y_max)
    # x_values = np.arange(x_min, x_max + 1, 1)  # step size of 1 for x
    # y_values = np.arange(y_min, y_max + 1, 1)  # step size of 1 for y
    
    # xx, yy = np.meshgrid(x_values, y_values)

    # coords = np.vstack([xx.ravel(), yy.ravel()]).T
    

    # change the points to homogeneous coordinates 
    coords_with_ones = np.hstack([interior_pts, np.ones((interior_pts.shape[0], 1))])
    print(coords_with_ones)
    
    # tranform each coordinate:
    
    new_pts = np.matmul(H, coords_with_ones.T).T
    new_pts = new_pts[:, :2] / new_pts[:, 2:]
    
    # print(np.size(coords_with_ones))
    # print(np.size(H))
    # raise NotImplementedError("warps_pts() is not implemented!")

    return new_pts
