import math
import numpy as np
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.utils.model_zoo as model_zoo


__all__ = ['fractallnext', 'fanext50', 'fanext101',
           'fanext152']


class FANeXtBottleneck(nn.Module):
    expansion = 2
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(FANeXtBottleneck, self).__init__()
        self.group_planlist = [2**int(np.log2(planes)-6)]+\
                [2**i for i in range(int(np.log2(planes)-6),int(np.log2(planes)-3))]
        self.groups = 8
        self.group_plan = self.group_planlist * self.groups
        
        '''
        #Original ResNeXt Structure
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, groups=32, stride=stride, 
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 2)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        '''
        
        
        self.conv1 = nn.Conv2d(inplanes,planes,kernel_size=1,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        for idx,planval in enumerate(self.group_plan):
            #exec('self.conv1_{idx}=nn.Conv2d(inplanes, {val}, kernel_size=1, bias=False)'.format(idx=idx,val=planval))
            #exec('self.bn1_{idx}=nn.BatchNorm2d({val})'.format(idx=idx,val=planval))
            #exec('self.conv2_{idx}=nn.Conv2d({val}, {val}, kernel_size=3,stride=stride,padding=1,bias=False)'\
            #     .format(idx=idx,val=planval))
            exec('self.conv2_{idx}=nn.Conv2d(planes, {val}, kernel_size=3, stride=stride, padding=1, bias=False)'\
                 .format(idx=idx,val=planval))
            
            #exec('self.bn2_{idx}=nn.BatchNorm2d({val})'.format(idx=idx,val=planval))
            #exec('self.conv3_{idx}=nn.Conv2d({val}, planes*2, kernel_size=1, bias=False)'.format(idx=idx,val=planval))
        
        #self.conv2_0 = nn.Conv2d(planes, planes/2, kernel_size=3,stride=stride,padding=1,bias=False)
        #self.conv2_1 = nn.Conv2d(planes, planes/2, kernel_size=3,stride=stride,padding=1,bias=False)
        
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=False)
        
        self.bn3 = nn.BatchNorm2d(planes * 2)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):    
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)
            
        finalout = residual
        
        '''
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        finalout = finalout + out
        
        finalout = self.relu(finalout)
        return finalout
        '''
        
        # ResistOOM
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        s = ','.join(['self.conv2_{idx}(out)'.format(idx=i) for i in range(32)])
        exec('out = torch.cat([{s}],1)'.format(s=s))
        
        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        
        finalout = finalout+out
        
        finalout = self.relu(finalout)
        return finalout
        
        
        '''
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
        '''

class FANeXtBottleneckV2(nn.Module):
    expansion = 2
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(FANeXtBottleneckV2, self).__init__()
        self.group_planlist = [2**int(np.log2(planes)-6)]+\
                [2**i for i in range(int(np.log2(planes)-6),int(np.log2(planes)-3))]
        self.groups = 8
        self.group_plan = self.group_planlist * self.groups
        
        '''
        #Original ResNeXt Structure
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, groups=32, stride=stride, 
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 2)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        '''
        
        
        self.conv1 = nn.Conv2d(inplanes,planes,kernel_size=1,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        for idx,planval in enumerate(self.group_planlist):
            #exec('self.conv1_{idx}=nn.Conv2d(inplanes, {val}, kernel_size=1, bias=False)'.format(idx=idx,val=planval))
            #exec('self.bn1_{idx}=nn.BatchNorm2d({val})'.format(idx=idx,val=planval))
            #exec('self.conv2_{idx}=nn.Conv2d({val}, {val}, kernel_size=3,stride=stride,padding=1,bias=False)'\
            #     .format(idx=idx,val=planval))
            exec('self.conv2_{idx}=nn.Conv2d(planes, {val}, kernel_size=3, groups={groups}, stride=stride, padding=1, bias=False)'\
                 .format(idx=idx,val=planval*self.groups,groups=self.groups))
            
            #exec('self.bn2_{idx}=nn.BatchNorm2d({val})'.format(idx=idx,val=planval))
            #exec('self.conv3_{idx}=nn.Conv2d({val}, planes*2, kernel_size=1, bias=False)'.format(idx=idx,val=planval))
        
        #self.conv2_0 = nn.Conv2d(planes, planes/2, kernel_size=3,stride=stride,padding=1,bias=False)
        #self.conv2_1 = nn.Conv2d(planes, planes/2, kernel_size=3,stride=stride,padding=1,bias=False)
        
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=False)
        
        self.bn3 = nn.BatchNorm2d(planes * 2)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):    
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)
            
        finalout = residual
        
        '''
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        finalout = finalout + out
        
        finalout = self.relu(finalout)
        return finalout
        '''
        
        # ResistOOM
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        s = ','.join(['self.conv2_{idx}(out)'.format(idx=i) for i in range(4)])
        exec('out = torch.cat([{s}],1)'.format(s=s))
        
        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        
        finalout = finalout+out
        
        finalout = self.relu(finalout)
        return finalout
        
        
        '''
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
        '''

class FANeXtBottleneckV3(nn.Module):
    expansion = 2
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(FANeXtBottleneckV3, self).__init__()
        self.group_planlist = [2**int(np.log2(planes)-6)]+\
                [2**i for i in range(int(np.log2(planes)-6),int(np.log2(planes)-3))]
        self.groups = 8
        self.group_plan = self.group_planlist * self.groups
        
        '''
        #Original ResNeXt Structure
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, groups=32, stride=stride, 
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 2)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        '''
        
        
        #self.conv1 = nn.Conv2d(inplanes,planes,kernel_size=1,bias=False)
        #self.bn1 = nn.BatchNorm2d(planes)
        for idx,planval in enumerate(self.group_planlist):
            exec('self.conv1_{idx}=nn.Conv2d(inplanes, {val}, kernel_size=1, bias=False)'.\
                 format(idx=idx,val=planval*self.groups))
            exec('self.bn1_{idx}=nn.BatchNorm2d({val})'.format(idx=idx,val=planval*self.groups))
            #exec('self.conv2_{idx}=nn.Conv2d({val}, {val}, kernel_size=3,stride=stride,padding=1,bias=False)'\
            #     .format(idx=idx,val=planval))
            exec('self.conv2_{idx}=nn.Conv2d({val}, {val}, kernel_size=3, groups={groups}, stride=stride, padding=1, bias=False)'\
                 .format(idx=idx,val=planval*self.groups,groups=self.groups))
            
            #exec('self.bn2_{idx}=nn.BatchNorm2d({val})'.format(idx=idx,val=planval))
            #exec('self.conv3_{idx}=nn.Conv2d({val}, planes*2, kernel_size=1, bias=False)'.format(idx=idx,val=planval))
        
        #self.conv2_0 = nn.Conv2d(planes, planes/2, kernel_size=3,stride=stride,padding=1,bias=False)
        #self.conv2_1 = nn.Conv2d(planes, planes/2, kernel_size=3,stride=stride,padding=1,bias=False)
        
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=False)
        
        self.bn3 = nn.BatchNorm2d(planes * 2)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):    
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)
            
        finalout = residual
        
        '''
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        finalout = finalout + out
        
        finalout = self.relu(finalout)
        return finalout
        '''
        
        # ResistOOM
        #out = self.conv1(x)
        #out = self.bn1(out)
        #out = self.relu(out)
        
        s = ','.join(['self.conv2_{idx}(self.relu(self.bn1_{idx}(self.conv1_{idx}(x))))'.format(idx=i) for i in range(4)])
        exec('out = torch.cat([{s}],1)'.format(s=s))
        
        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        
        finalout = finalout+out
        
        finalout = self.relu(finalout)
        return finalout
        
        
        '''
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
        '''
        
class FABigBlock(nn.Module):
    fracparam = 2
    def __init__():
        super(FAResNeXt, self).__init__()
    def _make_layer():
        layers = []
        #Ordinary, Make N Layer
        self.layers = layers
        return nn.Sequential(*layers)
    def forward(x):
        exec('level_0 = x')
        for i in range(len(self.layers)):
            idx = i+1
            exec('level_{idx} = self.layers[i](level_{i})'.format(idx=idx,i=i))
            '''
            tmp = block(self.inplanes, planes)
            tmpdist = self.fracparam
            while tmpdist <= len(layers):
                tmp = tmp + layers[-tmpdist]
                tmpdist *= self.fracparam
            layers.append(tmp)
            '''
        

        
class FAResNeXt(nn.Module):

    def __init__(self, block, layers, verticalfrac=False, fracparam=2, num_classes=1000):
        self.inplanes = 64
        self.fracparam = fracparam
        super(FAResNeXt, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        if verticalfrac:
            self._make_layer = self._make_fractal_layer
        else:
            self._make_layer = self._make_nonfractal_layer
            
        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 1024, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(1024 * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def _make_nonfractal_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        
        #print layers
        return nn.Sequential(*layers)
    
    def _make_fractal_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
            
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            tmp = block(self.inplanes, planes)
            tmpdist = self.fracparam
            while tmpdist <= len(layers):
                tmp = tmp + layers[-tmpdist]
                tmpdist *= self.fracparam
            layers.append(tmp)
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x    



def faresnext50(pretrained=False, **kwargs):
    """Constructs a ResNeXt-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FAResNeXt(FANeXtBottleneck, [3, 4, 6, 3], **kwargs)

    return model


def faresnext50v2(pretrained=False, **kwargs):
    """Constructs a ResNeXt-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FAResNeXt(FANeXtBottleneckV2, [3, 4, 6, 3], **kwargs)

    return model


def faresnext50v3(pretrained=False, **kwargs):
    """Constructs a ResNeXt-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FAResNeXt(FANeXtBottleneckV3, [3, 4, 6, 3], **kwargs)
    
    return model


def faresnext101(pretrained=False, **kwargs):
    """Constructs a ResNeXt-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FAResNeXt(FANeXtBottleneck, [3, 4, 23, 3], **kwargs)

    return model


def faresnext101v2(pretrained=False, **kwargs):
    """Constructs a ResNeXt-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FAResNeXt(FANeXtBottleneckV2, [3, 4, 23, 3], **kwargs)

    return model

def faresnext101v3(pretrained=False, **kwargs):
    """Constructs a ResNeXt-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FAResNeXt(FANeXtBottleneckV3, [3, 4, 23, 3], **kwargs)

    return model


def faresnext152(pretrained=False, **kwargs):
    """Constructs a ResNeXt-151 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FAResNeXt(FANeXtBottleneck, [3, 8, 36, 3], **kwargs)

    return model

