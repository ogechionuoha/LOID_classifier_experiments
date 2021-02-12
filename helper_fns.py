# helper functions
from sklearn.metrics import roc_curve, roc_auc_score, auc, precision_recall_fscore_support, confusion_matrix, accuracy_score

def add_pr_curve_tensorboard(writer, phase, targets, probs, preds, class_names, global_step=0):
    '''
    Takes in a target-class and plots the corresponding precision-recall curve
    '''
    tensorboard_preds = preds == targets
    tensorboard_probs = probs[:, targets]

    writer.add_pr_curve(f'{phase}/pr_curve/{class_names[targets]}', tensorboard_preds, tensorboard_probs, global_step=global_step)
    writer.flush()
    writer.close()
    
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    
def log_metrics(filename, y_target, y_pred, loss, saved=False, writer=None, epoch=0):
    
    with open(filename, 'a') as result:
        a, b = confusion_matrix(y_target, y_pred)
        tn, fp, fn, tp = *a, *b
        precision, recall, f1, support = precision_recall_fscore_support(y_target, y_pred)
        precision_0, precision_1, recall_0, recall_1, f1_0, f1_1, count_0, count_1 = *precision, *recall, *f1, *support
        accuracy = accuracy_score(y_target, y_pred)
        auroc = roc_auc_score(y_target, y_pred)
        line = ",".join([str(loss), str(accuracy), str(tn), str(fp), str(fn), str(tp), str(precision_0), str(precision_1), str(recall_0), str(recall_1), str(f1_0), str(f1_1), str(count_0), str(count_1), str(auroc), str(saved)+'\n'])
        result.write(line)
        
        if writer is not None:
            paths = filename.strip('.csv').split('/')
            phase = paths[-1]
            writer.add_scalar(phase+'/tn', tn, epoch)
            writer.add_scalar(phase+'/tp', tp, epoch)
            writer.add_scalar(phase+'/fn', fn, epoch)
            writer.add_scalar(phase+'/fp', fp, epoch)
            writer.add_scalar(phase+'/auroc', auroc, epoch)
            writer.flush()
            writer.close()

