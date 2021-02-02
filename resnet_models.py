import torch.nn as nn
from torchvision.models import resnet50

class ResNet50_GradCam(nn.Module):
    def __init__(self,num_classes = 2, finetune=False, visualise=False, pretrained=True):
        super(ResNet50_GradCam, self).__init__()

        self.visualise = visualise

        model = resnet50(pretrained=pretrained)
        
        freeze = True if not finetune else visualise
        
        #freezes weights
        #finetune=true, visual=false
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


