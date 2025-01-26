from lse import least_squares_estimation
import numpy as np

def ransac_estimator(X1, X2, num_iterations=60000):
    
    sample_size = 8

    eps = 10**-4

    best_num_inliers = -1
    best_inliers = None
    best_E = None

    for i in range(num_iterations):
        # permuted_indices = np.random.permutation(np.arange(X1.shape[0]))
        permuted_indices = np.random.RandomState(seed=(i*10)).permutation(np.arange(X1.shape[0]))
        sample_indices = permuted_indices[:sample_size]
        test_indices = permuted_indices[sample_size:]


        # Get the 8 sample points for E estimation
        X1_sample = X1[sample_indices]
        X2_sample = X2[sample_indices]

        # Step 2: Estimate E from the sample
        E = least_squares_estimation(X1_sample, X2_sample)
        e3 = [[0,-1,0],[1,0,0],[0,0,0]]
        
        
        # Step 3: Compute epipolar errors for test points and identify inliers
        inliers = list(sample_indices)
        for j in test_indices:
            x1 = X1[j]  # Column vector for point in first image
            x2 = X2[j]  # Column vector for point in second image

            d1 = ((x2 @ E @ x1) ** 2)/ (np.linalg.norm(e3 @ E @ x1)**2)
            d2 = ((x1 @ E.T @ x2.T) ** 2)/ (np.linalg.norm(e3 @ E.T @ x2.T)**2)

            # Check if the error is within the threshold
            if d1 + d2 < eps:
                inliers.append(j)




        # Convert inliers to a numpy array
        inliers = np.array(inliers)      

        if inliers.shape[0] > best_num_inliers:
            best_num_inliers = inliers.shape[0]
            best_E = E
            best_inliers = inliers


    return best_E, best_inliers