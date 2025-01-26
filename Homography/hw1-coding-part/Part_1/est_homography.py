import numpy as np

def est_homography(X, Y):
    """
    Calculates the homography of two planes, from the plane defined by X
    to the plane defined by Y. In this assignment, X are the coordinates of the
    four corners of the soccer goal while Y are the four corners of the penn logo

    Input:
        X: 4x2 matrix of (x,y) coordinates of goal corners in video frame
        Y: 4x2 matrix of (x,y) coordinates of logo corners in penn logo
    Returns:
        H: 3x3 homogeneours transformation matrix s.t. Y ~ H*X

    To compute the unknown matrix H we will use SVD 
    
    """
    # lamda *  Y = H * X
       
    # A[i][i+1] = [[-X[0][0], -X[0][1], -1, 0, 0, 0, Y[0][0] * X[0][0], Y[0][0] * X[0][1],  Y[0][0]],
    #           [0, 0,  0, -X[0][0], -X[0][1], -1, Y[0][1] * X[0][0], Y[0][1] *X[0][1], Y[0][1]]]  
    X = np.asarray(X)
    Y = np.asarray(Y)
    A= np.zeros((8,9))
    
    for i in range(4):
        A[2 * i] = [-X[i][0], -X[i][1], -1, 0, 0, 0, Y[i][0] * X[i][0], Y[i][0] * X[i][1],  Y[i][0]] 
        A[(2 * i) + 1]= [0, 0,  0, -X[i][0], -X[i][1], -1, Y[i][1] * X[i][0], Y[i][1] *X[i][1], Y[i][1]] 
        
        
    # print(A)
    
    [U,S,V]= np.linalg.svd(A)
    
    # print(np.round(V,-1))
    
    H = V[-1]. reshape(3,3)
    
    H[0][0]= V[-1][0]
    H[0][1]= V[-1][1]
    H[0][2]= V[-1][2]
    
    # 1,0 1,1 1,2 
    H[1][0] = V[-1][3]
    H[1][1] = V[-1][4]
    H[1][2] = V[-1][5]
    H[2][0] = V[-1][6]
    H[2][1] = V[-1][7]
    H[2][2] = V[-1][8]
    
    # print(np.size(H))  
    
    return H


if __name__ == "__main__":
    # You could run this file to test out your est_homography implementation
    #   $ python est_homography.py
    # Here is an example to test your code, 
    # but you need to work out the solution H yourself.
    X = np.array([[0, 0],
                  [0, 10], 
                  [5, 0], 
                  [5, 10]])
    
    Y = np.array([[3, 4], 
                  [4, 11],
                  [8, 5], 
                  [9, 12]])
    
    
    H = est_homography(X, Y)
    print(H)
    