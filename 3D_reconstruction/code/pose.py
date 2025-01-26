import numpy as np

def pose_candidates_from_E(E):
    transform_candidates = []
    
    # Define R_Z(pi/2) and R_Z(-pi/2)
    RZ_pos = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    RZ_neg = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    
    # SVD decomposition of E
    U, _, Vt = np.linalg.svd(E)
    
    # Get third column of U (T options)
    T_pos = U[:, 2]
    T_neg = -U[:, 2]

    # Four possible transformations with determinant check
    # (a) T = U[:,2], R = U * RZ(pi/2) * Vt
    R1 = U @ RZ_pos.T @ Vt
    
    if np.linalg.det(R1) < 0:
        R1 = -R1  # Flip to ensure det(R) = +1
    transform_candidates.append({"T": T_pos, "R": R1})
    
    # (b) T = U[:,2], R = U * RZ(-pi/2) * Vt
    R2 = U @ RZ_neg.T @ Vt
    
    if np.linalg.det(R2) < 0:
        R2 = -R2  # Flip to ensure det(R) = +1
    transform_candidates.append({"T": T_pos, "R": R2})
    
    # (c) T = -U[:,2], R = U * RZ(pi/2) * Vt
    R3 = U @ RZ_pos.T @ Vt
    
    if np.linalg.det(R3) < 0:
        R3 = -R3  # Flip to ensure det(R) = +1
    transform_candidates.append({"T": T_neg, "R": R3})
    
    # (d) T = -U[:,2], R = U * RZ(-pi/2) * Vt
    R4 = U @ RZ_neg.T @ Vt
    
    if np.linalg.det(R4) < 0:
        R4 = -R4  # Flip to ensure det(R) = +1
    transform_candidates.append({"T": T_neg, "R": R4})
    
    return transform_candidates
