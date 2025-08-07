import torch.nn as nn
from .proxyless_nets import ProxylessNASNets
from ..modules import *
from ..modules.layers import build_activation
from ..utils import val2list

__all__ = ['ResNet18']


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, act_func='relu'):
        super(BasicBlock, self).__init__()
        
        self.conv1 = ConvLayer(
            in_channels, out_channels, kernel_size=3, stride=stride, 
            use_bn=True, act_func=act_func, ops_order='weight_bn_act'
        )
        self.conv2 = ConvLayer(
            out_channels, out_channels, kernel_size=3, stride=1,
            use_bn=True, act_func='none', ops_order='weight_bn_act'
        )
        
        self.downsample = downsample
        self.act_func = build_activation(act_func, inplace=True)
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.conv2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.act_func(out)
        
        return out


class ResNet18(ProxylessNASNets):

    def __init__(self, n_classes=1000, dropout_rate=0., act_func='relu', img_channel=3, pretrained=True):
        
        # ResNet18 architecture: [2, 2, 2, 2] blocks for each layer
        layers = [2, 2, 2, 2]
        channels = [64, 128, 256, 512]
        
        act_func = val2list(act_func, 1 + len(layers) + 1)  # first_conv + layers + classifier
        
        # First conv layer
        first_conv = ConvLayer(
            img_channel, 64, kernel_size=7, stride=2, use_bn=True, 
            act_func=act_func[0], ops_order='weight_bn_act'
        )
        
        # Max pooling after first conv
        maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Build ResNet blocks
        blocks = []
        in_channels = 64
        
        for layer_idx, (num_blocks, out_channels) in enumerate(zip(layers, channels)):
            stride = 1 if layer_idx == 0 else 2
            
            # First block may need downsampling
            downsample = None
            if stride != 1 or in_channels != out_channels * BasicBlock.expansion:
                downsample = ConvLayer(
                    in_channels, out_channels * BasicBlock.expansion,
                    kernel_size=1, stride=stride, use_bn=True, act_func='none'
                )
            
            # First block of the layer
            blocks.append(BasicBlock(
                in_channels, out_channels, stride, downsample, act_func[layer_idx + 1]
            ))
            in_channels = out_channels * BasicBlock.expansion
            
            # Remaining blocks
            for _ in range(1, num_blocks):
                blocks.append(BasicBlock(
                    in_channels, out_channels, act_func=act_func[layer_idx + 1]
                ))
        
        # Global average pooling and classifier
        feature_mix_layer = None  # ResNet uses global average pooling directly
        classifier = LinearLayer(512 * BasicBlock.expansion, n_classes, dropout_rate=dropout_rate)
        
        super(ResNet18, self).__init__(first_conv, blocks, feature_mix_layer, classifier)
        
        # Store maxpool separately as it's not part of the base class structure
        self.maxpool = maxpool
        
        # Load pretrained weights if requested
        if pretrained:
            self._load_pretrained_weights()

    def _load_pretrained_weights(self):
        """Load pretrained ResNet18 weights from torchvision"""
        try:
            import torchvision.models as models
            pretrained_model = models.resnet18(pretrained=True)
            
            # Load first conv weights
            self.first_conv.conv.weight.data.copy_(pretrained_model.conv1.weight.data)
            self.first_conv.bn.weight.data.copy_(pretrained_model.bn1.weight.data)
            self.first_conv.bn.bias.data.copy_(pretrained_model.bn1.bias.data)
            self.first_conv.bn.running_mean.data.copy_(pretrained_model.bn1.running_mean.data)
            self.first_conv.bn.running_var.data.copy_(pretrained_model.bn1.running_var.data)
            
            # Load maxpool weights (it's just a layer, no weights to copy)
            
            # Load block weights
            pretrained_layers = [pretrained_model.layer1, pretrained_model.layer2, 
                               pretrained_model.layer3, pretrained_model.layer4]
            
            block_idx = 0
            for layer_blocks in pretrained_layers:
                for pretrained_block in layer_blocks:
                    if block_idx < len(self.blocks):
                        our_block = self.blocks[block_idx]
                        
                        # Copy conv1 weights
                        our_block.conv1.conv.weight.data.copy_(pretrained_block.conv1.weight.data)
                        our_block.conv1.bn.weight.data.copy_(pretrained_block.bn1.weight.data)
                        our_block.conv1.bn.bias.data.copy_(pretrained_block.bn1.bias.data)
                        our_block.conv1.bn.running_mean.data.copy_(pretrained_block.bn1.running_mean.data)
                        our_block.conv1.bn.running_var.data.copy_(pretrained_block.bn1.running_var.data)
                        
                        # Copy conv2 weights
                        our_block.conv2.conv.weight.data.copy_(pretrained_block.conv2.weight.data)
                        our_block.conv2.bn.weight.data.copy_(pretrained_block.bn2.weight.data)
                        our_block.conv2.bn.bias.data.copy_(pretrained_block.bn2.bias.data)
                        our_block.conv2.bn.running_mean.data.copy_(pretrained_block.bn2.running_mean.data)
                        our_block.conv2.bn.running_var.data.copy_(pretrained_block.bn2.running_var.data)
                        
                        # Copy downsample weights if exists
                        if our_block.downsample is not None and pretrained_block.downsample is not None:
                            our_block.downsample.conv.weight.data.copy_(pretrained_block.downsample[0].weight.data)
                            our_block.downsample.bn.weight.data.copy_(pretrained_block.downsample[1].weight.data)
                            our_block.downsample.bn.bias.data.copy_(pretrained_block.downsample[1].bias.data)
                            our_block.downsample.bn.running_mean.data.copy_(pretrained_block.downsample[1].running_mean.data)
                            our_block.downsample.bn.running_var.data.copy_(pretrained_block.downsample[1].running_var.data)
                        
                        block_idx += 1
            
            # Load classifier weights (final fc layer)
            self.classifier.linear.weight.data.copy_(pretrained_model.fc.weight.data)
            self.classifier.linear.bias.data.copy_(pretrained_model.fc.bias.data)
            
            print("Loaded pretrained ResNet18 weights from torchvision")
            
        except ImportError:
            print("Warning: torchvision not available, using random weights")
        except Exception as e:
            print(f"Warning: Failed to load pretrained weights: {e}")

    def forward(self, x):
        x = self.first_conv(x)
        x = self.maxpool(x)
        
        for block in self.blocks:
            x = block(x)
        
        # Global average pooling
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        
        return x