import os
import torch
import torch.nn as nn
from torchvision.models import resnet50
from attention_module.models.model_resnet import ResidualNet

class ResNet50_GradCam(nn.Module):
    def __init__(self,num_classes = 2, finetune=False, visualise=False, pretrained=True):
        super(ResNet50_GradCam, self).__init__()

        self.visualise = visualise

        model = resnet50(pretrained=pretrained)
        
        freeze = True if not finetune else visualise

        if freeze:
            print("Freezing params...")
            for param in model.parameters():
                param.requires_grad = False

        self.base = nn.Sequential(*list(model.children())[:-2])

        self.avgpool = model.avgpool

        #final layer
        infeatures = model.fc.in_features
        self.fc = nn.Linear(infeatures, num_classes)

        # placeholder for the gradients
        self.gradients = None

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    # method for the activation exctraction
    def get_activations(self, x):
        return self.base(x)

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        #x.requires_grad = True
        x = self.base(x)

        # register the hook if you want to visualise the intermediate layers
        if self.visualise:
            h = x.register_hook(self.activations_hook)

        # apply the remaining pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out


    
class ResNet50_CBAM(ResNet50_GradCam):
    def __init__(self,num_classes = 2, finetune=False, visualise=False, pretrained=True, attn_type=None):
        super(ResNet50_CBAM, self).__init__(num_classes = 2, finetune=finetune, visualise=visualise, pretrained=pretrained)

        pretained_cbam_path = './attention_module/RESNET50_CBAM.pth'
        
        uses_cbam = '' if attn_type==None else 'with cbam' 

        if pretrained and not finetune:
            print(f'Using resnet 50 pretrained model {uses_cbam} for feature extraction')
        elif pretrained and finetune:
            print(f'Using resnet 50 pretrained model {uses_cbam} for finetuning')

        get_pretrained = False if attn_type and pretrained else pretrained
        model = ResidualNet('ImageNet', 50, 1000, attn_type, pretrained=get_pretrained, modelpath=None)

        if attn_type is not None and pretrained==True:
            if os.path.isfile(pretained_cbam_path):
                print("=> loading checkpoint '{}'".format(pretained_cbam_path))
                checkpoint = torch.load(pretained_cbam_path)
                for key in list(checkpoint['state_dict'].keys()):
                    newkey = key.replace('module.','')
                    checkpoint['state_dict'][newkey] = checkpoint['state_dict'].pop(key)
                
                model.load_state_dict(checkpoint['state_dict'])
                print("Loaded checkpoint !!")
            else:
                print("=> no checkpoint found at '{}'".format(pretained_cbam_path))
                
        self.base = nn.Sequential(*list(model.children())[:-2])

        self.avgpool = model.avgpool

        #final layer
        infeatures = model.fc.in_features
        self.fc = nn.Linear(infeatures, num_classes)







