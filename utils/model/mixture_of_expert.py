import torch
import torch.nn as nn
from feature_varnet import FeatureVarNet_sh_w as VarNet
from simple_classifier import SimpleClassifier

class MoEModel(nn.Module):
    def __init__(self, brain_model_args, knee_model_args):
        super().__init__()
        self.brain_net = VarNet(**brain_model_args)
        self.knee_net  = VarNet(**knee_model_args)
        self.classifier = SimpleClassifier()

    def load_models(self, brain_model_path, knee_model_path, classifier_path, device):
        brain_ckpt = torch.load(brain_model_path, map_location=device, weights_only=False)
        self.brain_net.load_state_dict(brain_ckpt['model'])

        knee_ckpt = torch.load(knee_model_path, map_location=device, weights_only=False)
        self.knee_net.load_state_dict(knee_ckpt['model'])

        classifier_ckpt = torch.load(classifier_path, map_location=device, weights_only=False)
        self.classifier.load_state_dict(classifier_ckpt['model'])

    def forward(self, masked_kspace, mask, input_img):
        '''
        Forward pass through the MoE model.
        masked_kspace: Input k-space tensor (1, C, H, W, 2)
        mask: Mask tensor (1, C, H, W)
        input_img: Input image tensor (1, C, H, W)

        Returns:
            out: Reconstructed MRI image after processing through the network. (1, H, W)
        '''
        input_img = input_img
        logits = self.classifier(input_img)
        choice = torch.argmax(logits, dim=1)        # 0: brain, 1: knee

        if choice.item() == 0:
            out = self.brain_net(masked_kspace, mask)
        else:
            out = self.knee_net(masked_kspace, mask)

        return out