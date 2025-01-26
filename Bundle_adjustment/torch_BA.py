import torch
import numpy as np
from batch_rodrigue import batch_rodrigues

class torch_BA(torch.nn.Module):
    
    def __init__(self, F:int, N:int, features:torch.Tensor, device:str, lr:float) -> None:
        """
        The initial of torch BA module
        steps: optimized steps
        F: represents the target frame number
        N: represents number of 3D features/ point
        features: [N, F, 2] eg (x, y) calibrated feature point x and y in each frame
        """
        
        super().__init__()
        self.N = N 
        self.F = F
        self.features = features.to(device)
        self.device = device
        assert features.shape[0] == N and features.shape[1] == F and features.shape[2] == 2

        #############################  TODO 4.1 BEGIN  ############################
        # DO NOT CHANGE THE NAME, BECAUSE IT FILLS IN THE OPIMIZOR
        # theta (0,0,0) is for one frame- so for 5 it is 5 x 3, I think its 4 x 3 as 5 frames.?
        
        self.theta = torch.nn.Parameter(torch.zeros((self.F -1 , 3), device=device),requires_grad=True)
        self.trans = torch.nn.Parameter(torch.zeros((self.F -1, 3), device=device), requires_grad=True)
        self.key3d = torch.nn.Parameter(torch.ones((self.N ,3), device= device)+ torch.tensor([0, 0, 1] ,device=device),requires_grad=True) 

        #############################  TODO 4.1 END  ##############################

        self.config_optimizer(lr = lr)

    def forward_one_step(self):
        """
        Forward function running one opimization
        """
        self.optimizer.zero_grad()
        loss = self.compute_loss()
        loss.backward() 
        self.optimizer.step()
        return loss.item()
    
    def reprojection(self, training = True):
        """
        Reproject 3d keypoints to each frame
        You can use provided function batch_rodrigues to convert axis angle representation to rotation matrix
        """
        
        # what is this?
        device = self.device


        # the original N features, converted to N x F features(project to each view) after reprojection
        # project 3D features back to their 2D frames - then calculate error
        # xcamera ij  = Rj Xworld i + tj  
        # batch processing
        
        # each theta is concatenated row-wise wrt world frame we chose, and then R is found 
        # how is each theta filled? We chose it as 0 too
        world_frame_theta = torch.zeros((1,3),   device = self.device)
        rot_matrix = torch.cat([self.theta, world_frame_theta], dim = 0)
        rotations = batch_rodrigues(rot_matrix) # F , 3 , 3 
        
        world_frame_trans = torch.zeros((1,3),  device = self.device)
        trans_matrix = torch.cat([self.trans, world_frame_trans], dim = 0) # F, 3
        
        # key_3d is the world features. - convert to homgenoues 
        homo_for_batching = torch.ones((self.key3d.shape[0], 1), device=self.device)
        key_3d_homo = torch.cat([self.key3d, homo_for_batching], dim = -1) # Shape: (N, 4)
        key_3d_homo = key_3d_homo.T.unsqueeze(0).expand(rotations.shape[0], -1, -1)  # Shape: (F, 4, N)

        # Perform batch matrix multiplication: rotations (F, 3, 3) and key_3d_homo (F, 4, N)
        # We need to select the first three elements from key_3d_homo (F, 3, N)
        key_3d_homo = key_3d_homo[:, :3, :]  # Slice key_3d_homo to (F, 3, N)


        # Compute camera coordinates
        x_cam = torch.bmm(rotations, key_3d_homo) + trans_matrix.unsqueeze(-1)  # Shape: (F, 3, N)

        #normalization of reprojected features
        reproj_features = x_cam[:, :2, :] / x_cam[:, 2:3, :]  # Shape: (F, 2, N)
        
        if training == False:
            reproj_features = reproj_features.detach().cpu()



                        
        print("rotations shape:", rotations.shape)
        print("key_3d_homo shape:", key_3d_homo.shape)
        print("x_cam shape:", x_cam.shape)
        print("reproj_features shape:", reproj_features.shape)

        return reproj_features.permute(2,0,1)  # Return normalized calibrated keypoint in (N,F,2), with z = 1 ignored


    def config_optimizer(self, lr):
        self.optimizer = torch.optim.Adam([self.theta, self.trans, self.key3d],lr=lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[5], gamma=0.1)

    def compute_loss(self):
        """
        Error computation function
        The error is defined as the square of the distance between reprojected calibrated points (x', y')
        and given calibrated points (x, y)
        """
        # Predicted values (reprojected features)
        reproj_features = self.reprojection()

        # Ground truth target values
        target_features = self.features
        
        print("reproj_features shape in this function:", reproj_features.shape)
        print("target_features shape before:", target_features.shape)


        # Ensure target_features has the correct shape N, F, 2
        target_features = target_features.permute(1, 2, 0)  # Change shape to [5, 2, 332]
        print("target_features shape after:", target_features.shape)


        # Compute the loss (mean squared error)
        loss = torch.mean((reproj_features - target_features) ** 2)
        return loss




    def scheduler_step(self):
        self.scheduler.step()
        
    def reset_parameters(self):
        self.__init__(F=self.F, N=self.N, features=self.features)

    def save_parameters(self, to_rotm = True):
        theta = self.theta
        full_theta = torch.cat([theta, torch.zeros((1,3),device=self.device)], dim=0)
        if to_rotm:
            theta = batch_rodrigues(self.theta)
            full_theta = torch.cat([theta, torch.eye(3,device=self.device).unsqueeze(0)], dim=0)

        trans = torch.cat([self.trans, torch.zeros((1,3),device=self.device)], dim=0)
        return full_theta.detach().cpu(), trans.detach().cpu(), self.key3d.detach().cpu()
