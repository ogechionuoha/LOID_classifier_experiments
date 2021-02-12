import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.metrics import roc_curve, roc_auc_score, auc, precision_recall_fscore_support, confusion_matrix, accuracy_score


def load_model(model, checkpoint, device):
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    model.to(device)
    return model
    
def get_image_results(model, image_file):
    img = plt.imread(image_file)
    img = transforms.ToPILImage()(img)
    img = transformations(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    res = model.forward(img)
    pred = res.argmax(dim=1)

    res[:,pred.item()].backward()

    return img, res, pred

def get_heatmap(model, img, show_heatmap=False):
    # pull the gradients out of the model
    gradients = model.get_activations_gradient()

    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # get the activations of the last convolutional layer
    activations = model.get_activations(img).detach()

    # weight the channels by corresponding gradients
    for i in range(512):
        activations[:, i, :, :] *= pooled_gradients[i]

    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()

    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = heatmap.cpu()

    heatmap = np.maximum(heatmap, 0)

    # normalize the heatmap
    heatmap /= torch.max(heatmap)
    
    if show_heatmap:
        # draw the heatmap
        plt.matshow(heatmap.squeeze())

    heatmap = heatmap.numpy()
    
    return heatmap

def generate_superimposition(image_file, heatmap, outputs='./outputs'):
    if not os.path.exists(outputs):
        os.mkdir(outputs)
        
    res_folder = outputs +'/'+model_short
    
    filename = image_file.split('/')[-1]
    res_file = os.path.join(res_folder, filename.split('.jpg')[0]+'_'+model_short+'.jpg')
    img = cv2.imread(image_file)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    cv2.imwrite(res_file, superimposed_img)
    return image_file,res_file
    
def show_superimposition(image_file, res_file):
    rimg = plt.imread(res_file)
    oimg = plt.imread(image_file)
    fig, ax = plt.subplots(1,2,figsize=(10,10))
    plt.figure(figsize=(20,20))
    ax[0].imshow(oimg)
    ax[1].imshow(rimg)
    plt.show()
    
def get_gradcam(model, image_file, outputs = './outputs'):
    img, res, pred = get_image_results(model, image_file)
    heatmap = get_heatmap(model, img)
    image_file, res_file = generate_superimposition(image_file, heatmap, outputs)
    return image_file, res_file

def show_superimpositions(image_pairs, title="Gradcam results"):
    fig, ax = plt.subplots(len(image_pairs),2,figsize=(90,90))
    row = 0
    for image_file, res_file in image_pairs:
        oimg = plt.imread(image_file)
        rimg = plt.imread(res_file)
        
        
        ax[row, 0].imshow(oimg)
        ax[row, 1].imshow(rimg)
        
        row+=1
    plt.tight_layout() 
    plt.suptitle(title)
    plt.show() 