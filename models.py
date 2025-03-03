import torch
from torchvision import transforms
import torchvision.models as tmodels
    
class Inception(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels, mid_kernel = [3, 5, 7], stride=1):
        """
        Initializes a new Inception block composed of several parallel convolution blocks
        that extract features with different scales and neighbourhood sizes
        
        Args:
            in_channels (int): the number of input features of given layer
            out_channels (int): the number of output features of given layer
            mid_kernel (list of ints): sizes of middle convolution kernels of inception module
        """
        
        super().__init__()
        
        self.branch_left = torch.nn.Conv2d(in_channels, out_channels, 1, stride)
        self.branch_1 = torch.nn.Conv2d(in_channels, out_channels, mid_kernel[0], stride, padding=1)
        self.branch_2 = torch.nn.Conv2d(in_channels, out_channels, mid_kernel[1], stride, padding=2)
        self.branch_right = torch.nn.Conv2d(in_channels, out_channels, mid_kernel[2], stride, padding=3)
    
    def forward(self, x):    
        branch_left = torch.nn.functional.leaky_relu(self.branch_left(x))
        branch_1 = torch.nn.functional.leaky_relu(self.branch_1(x)) 
        branch_2 = torch.nn.functional.leaky_relu(self.branch_2(x)) 
        branch_right = torch.nn.functional.leaky_relu(self.branch_right(x)) 

        out = torch.cat((branch_left, branch_1, branch_2, branch_right), dim=1)
        
        return out
    
class ResBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, activation=True):
        """
        Initializes a new Residual Block object
        
        h(x) = f(x) + x
        
        Args:
            in_channels (int): the number of input features of given layer
            out_channels (int): the number of output features of given layer,
                for residual block it's recommended to change channel size together with activation map size (size of conv output)
        """
        super(ResBlock, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.downsample = None
        self.activation = activation
        
        if stride > 1:
            self.downsample = torch.nn.Conv2d(in_channels, out_channels, 1, stride=2)
            
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, 3, stride=stride, padding=1)
        self.norm1 = torch.nn.InstanceNorm2d(out_channels)
        self.norm2 = torch.nn.InstanceNorm2d(out_channels)
        
    def forward(self, x):
        
        if self.downsample:
            residual = self.downsample(x)
        else:
            residual = x
       
        x = self.conv1(x)
        x = self.norm1(x)
        
        if self.activation:
            x = torch.nn.functional.leaky_relu(x)
            
        x = self.conv2(x)
        
        if self.activation:
            x = torch.nn.functional.leaky_relu(x)
        
        x = x + residual
        x = self.norm2(x)
        
        if self.activation:
            x = torch.nn.functional.leaky_relu(x)
        
        return x
 
class Encoder(torch.nn.Module):
    def __init__(self):
        """
        Initializes a new Encoder object that encodes input image into latent space (compressed feature representation)
        """
        super().__init__()
        
        self.downsample = torch.nn.MaxPool2d(2, 2) 
        
        self.incept1 = Inception(1, 16, [3, 5, 7], 2)
        
        self.res1 = ResBlock(64, 128, 2)
        self.res2 = ResBlock(128, 128, 2)
        self.res3 = ResBlock(128, 128, 2)
        self.res4 = ResBlock(128, 128, 2)
        self.res5 = ResBlock(128, 128, 2)
        self.res6 = ResBlock(128, 128, 2)
        self.res7 = ResBlock(128, 128, 2)
        self.res8 = ResBlock(128, 64, 2, False)
        
    def forward(self, x):
        
        x = self.incept1(x)
        
        x = self.res1(x) 
        x = self.res2(x) + self.downsample(x)
        x = torch.nn.functional.dropout2d(x, p=0.4)
        x = self.res4(x) + self.downsample(x)
        x = torch.nn.functional.dropout2d(x, p=0.4)
        x = self.res5(x) + self.downsample(x)
        x = torch.nn.functional.dropout2d(x, p=0.4)
        x = self.res6(x) + self.downsample(x)
        x = torch.nn.functional.dropout2d(x, p=0.4)
        x = self.res7(x) + self.downsample(x)
        x = torch.nn.functional.dropout2d(x, p=0.5)
        x = self.res8(x)
        
        x = torch.flatten(x)
        
        return x

class GRUAggregation(torch.nn.Module):
    
    def __init__(self, embedding_size, heads=2):
        super().__init__()
        
        self.gru = torch.nn.GRU(embedding_size, embedding_size, heads)
        
    def forward(self, x) -> torch.Tensor:
        x = x.float()
        
        output, hn = self.gru(x)
        output_aggr, hn_aggr = output.sum(dim=0), hn.sum(dim=0)
        
        return output_aggr, hn_aggr

class MLP(torch.nn.Module):
    def __init__(self, out_channels=1):
        super(MLP, self).__init__()

        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 32)
        self.lin3 = torch.nn.Linear(32, 16)
        self.lin4 = torch.nn.Linear(16, out_channels)

    def forward(self, x):

        x = self.lin1(x)
        x = torch.nn.functional.relu(x)
        x = self.lin2(x)
        x = torch.nn.functional.relu(x)
        x = self.lin3(x)
        x = torch.nn.functional.relu(x)
        x = self.lin4(x)

        return x