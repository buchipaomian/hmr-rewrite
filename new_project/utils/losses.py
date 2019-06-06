"""
Module for various loss functions
"""

import torch
import torch.nn as nn

class UnetLoss(nn.Module):
    def __init__(self,device = torch.device('cpu')):
        super(UnetLoss,self).__init__()
        self.loss_function = nn.BCELoss()
        self.device = device
    def forward(self,model,point):
        target = torch.zeros(112,112,112)
        x,y,z = 0,0,0
        for count,i in enumerate(point):
            if count%3 == 0:
                x = (i//2+112)//2
            elif count%3 == 1:
                y = (i//2+112)//2
            else:
                z = (i//2+112)//2
                try:
                    target[x][y][z] = 1
                except:
                    continue
        target = target.to(self.device)
        loss = self.loss_function(model,target)
        return loss
        
            

# class GANLoss(nn.Module):
#     def __init__(self, mse=True):
#         super(GANLoss, self).__init__()
#         if mse:
#             self.loss_function = nn.MSELoss()
#         else:
#             self.loss_function = nn.BCELoss()

#     def forward(self, logit, is_real):
#         if is_real:
#             target = torch.ones_like(logit)
#         else:
#             target = torch.zeros_like(logit)

#         return self.loss_function(logit, target)
