import numpy as np
import matplotlib.pyplot as plt

def show_reprojections(image1, image2, uncalibrated_1, uncalibrated_2, P1, P2, K, T, R, plot=True):

  """ YOUR CODE HERE
  """
  # Transform P2 to the coordinate system of Camera 1 and project it onto Camera 1's image plane
  print(P2.shape)
  P2_in_cam1 = R.T @ (P2.T - T.reshape(-1, 1) ) # going reverse, coordinate system changes. 
  P2proj = (K @ P2_in_cam1).T
  
  # Transform P1 to the coordinate system of Camera 2 and project it onto Camera 2's image plane
  P1_in_cam2 = R @ P1.T + T.reshape(-1, 1)  #R is from p1 to p2 , 
  P1proj = (K @ P1_in_cam2).T


  """ END YOUR CODE
  """

  if (plot):
    plt.figure(figsize=(6.4*3, 4.8*3))
    ax = plt.subplot(1, 2, 1)
    ax.set_xlim([0, image1.shape[1]])
    ax.set_ylim([image1.shape[0], 0])
    plt.imshow(image1[:, :, ::-1])
    plt.plot(P2proj[:, 0] / P2proj[:, 2],
           P2proj[:, 1] / P2proj[:, 2], 'bs')
    plt.plot(uncalibrated_1[0, :], uncalibrated_1[1, :], 'ro')

    ax = plt.subplot(1, 2, 2)
    ax.set_xlim([0, image1.shape[1]])
    ax.set_ylim([image1.shape[0], 0])
    plt.imshow(image2[:, :, ::-1])
    plt.plot(P1proj[:, 0] / P1proj[:, 2],
           P1proj[:, 1] / P1proj[:, 2], 'bs')
    plt.plot(uncalibrated_2[0, :], uncalibrated_2[1, :], 'ro')
    
  else:
    return P1proj, P2proj