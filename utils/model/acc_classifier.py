import torch

class acc_classifier():
    def __init__(self, acc_threshold=0.25):
        self.acc_threshold = acc_threshold

    def __call__(self, masked_kspace):
        """
        Classify the acceleration factor based on the k-space data and mask.
        
        Args:
            kspace (torch.Tensor): K-space data.
            mask (torch.Tensor): Mask applied to the k-space data.
        
        Returns:
            int: Acceleration factor (4 or 8).
        """
        # Calculate the number of non-zero elements in the masked k-space
        num_nonzero = torch.count_nonzero(masked_kspace)
        
        # Determine the acceleration factor based on the threshold
        if num_nonzero > self.acc_threshold * masked_kspace.numel():
            return 4
        else:
            return 8