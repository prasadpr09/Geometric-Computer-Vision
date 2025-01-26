import numpy as np

def least_squares_estimation(X1, X2):
  """ 
  Input - N x 3 Matrices, Calibrated points as inputs
  
  Output - Estimates E using SVD 
  
  Don’t forget to project the E you obtain onto the space of essential matrices by
  redecomposing [U,S,V] = svd(E) and returning Udiag(1, 1, 0)V
  
  """
  # p = X1  # (393 x 3) 
  # q = X2  # (393 x 3)
  n = X1.shape[0]
  a =np.zeros((n,9)) # 8 x 9 ? nah its 393 x 9
              #e       9 x 1
  

  for i in range(n):
      x1, y1, z1 = X1[i]
      x2, y2, z2 = X2[i]
      
      a[i] = [x2 * x1, x2 * y1, x2,
              y2 * x1, y2 * y1, y2,
              x1 , y1 , 1]

  u,s,vt = np.linalg.svd(a)
  
  # Last row of V^T (smallest singular value) gives the solution for e
  e = vt[-1]
  
  E = e.reshape(3,3)
  
  # make E valid- Project E onto the space of essential matrices
  
  U, S, Vt = np.linalg.svd(E)
  
  E = U @ np.diag([1, 1, 0]) @ Vt
  
  return E 
  
  # U,S,Vt = np.linalg(e)
  
  
  
  # return E
