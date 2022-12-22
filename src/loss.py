# Libraries
import torch

# Main classes
class PowerIoULoss(torch.nn.Module):
    def __init__(self, p: float=2, eps:float=1e-6, ignore_index:int=255):
        """
        Generalisation of both Dice and Jaccard loss functions.
        Reference: https://www.scitepress.org/Papers/2021/103040/103040.pdf
        """
        super(PowerIoULoss, self).__init__()
        self.p = p
        self.eps = eps
        self.ignore_index = ignore_index

    def forward(self, inputs:torch.Tensor, targets:torch.Tensor):
        # Get logits from model output
        inputs = inputs.get('logits') # shape (batch_size, num_classes, img_size/4, img_size/4)
        inputs = torch.nn.functional.interpolate(inputs, size=targets.shape[-2:], mode="bilinear", align_corners=False)

        # Flatten label and prediction tensors
        mask = targets.ne(self.ignore_index)
        inputs = torch.masked_select(inputs, mask).view(-1)
        targets = torch.masked_select(targets, mask).view(-1)

        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (torch.pow(inputs, self.p) + torch.pow(targets, self.p)).sum()
        IoU = (intersection + self.eps)/(total - intersection + self.eps)
        return 1 - IoU