import torch
import torch.nn as nn 
import torch.nn.functional as F
 
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        '''
        An Implementation of focalloss
        coder: Jeff pan
        Formula: -(alpha)*[(1-pt)**gamma]*log(pt) 
                alpha - balance quantity of different sample(only use to Two classification, the value is PositiveSample: NegtiveSample)
                gamma - to decrease the effect of the sample which are easy to classification
                pt    - probability describing the sample belong to correct class
        '''
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # create mask
        mask = torch.zeros_like(inputs)
        for i, v in enumerate(targets):
            mask[i][v] = 1
        mask = mask.bool()

        # get multiplication factor
        res_softmax = F.softmax(inputs, dim=1)
        # print(res_softmax)
        pt = torch.masked_select(res_softmax, mask)
        # print(pt)
        loss = -(self.alpha)*((1-pt)**self.gamma)*torch.log(pt)
        loss = torch.mean(loss)
        return loss

if __name__ == '__main__':
    inputs = torch.randn((8, 4))
    # print(inputs)
    targets = torch.tensor([0, 1, 2, 3, 1, 2, 2, 1])
    # print(targets)   
    a = FocalLoss()
    print(a(inputs, targets))