import torch
import torch.nn as nn 
import torch.nn.functional as F
 
class FocalLoss(nn.Module):
    def __init__(self, classnum=None, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        '''
        An Implementation of focalloss
        coder: Jeff pan
        Formula: -(alpha)*[(1-pt)**gamma]*log(pt) 
                alpha - balance quantity of different sample, for example: [10, 20, 30, 40], 10 means first class's quantity.
                gamma - to decrease the effect of the sample which are easy to classification
                pt    - probability describing the sample belong to correct class
        '''
        if not classnum: print("Enter parameters: classnum and alpha.\nUsage: loss_fc = FocalLoss(4, [10000, 20000, 20000, 9000])")
        self.alpha = sum(alpha)/torch.tensor(alpha, dtype=torch.float32)
        self.alpha = self.alpha/self.alpha.sum()
        # print(self.alpha)
        self.gamma = gamma

    def forward(self, inputs, targets):
        # create mask
        mask = torch.zeros_like(inputs)
        for i, v in enumerate(targets):
            mask[i][v] = 1
        mask = mask.bool()

        # get multiplication factor
        self.alpha = self.alpha.repeat(inputs.shape[0], 1)
        res_softmax = F.softmax(inputs, dim=1)
        self.alpha = torch.masked_select(self.alpha, mask)
        pt = torch.masked_select(res_softmax, mask)
        loss = -(self.alpha)*(torch.pow((1-pt), self.gamma))*torch.log(pt)
        loss = torch.mean(loss)
        return loss

if __name__ == '__main__':
    inputs = torch.randn((8, 4))
    # print(inputs)
    targets = torch.tensor([0, 1, 2, 3, 1, 2, 2, 1])
    # print(targets)   
    a = FocalLoss(4, [1000, 2000, 3000, 4000])
    print(a(inputs, targets))