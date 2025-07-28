import torch
import torch.nn as nn
from utils.model.feature_varnet import FeatureVarNet_sh_w as VarNet
from simple_classifier import SimpleClassifier
from acc_classifier import acc_classifier

class MoEModel(nn.Module):
    def __init__(self, brain_model_acc4_args, brain_model_acc8_args, knee_model_acc4_args, knee_model_acc8_args):
        super().__init__()
        self.brain_net_acc4 = VarNet(**brain_model_acc4_args)
        self.brain_net_acc8 = VarNet(**brain_model_acc8_args)
        self.knee_net_acc4 = VarNet(**knee_model_acc4_args)
        self.knee_net_acc8 = VarNet(**knee_model_acc8_args)
        self.classifier = SimpleClassifier()
        self.acc_classifier = acc_classifier()

    def load_models(self, brain_model_acc4_path, brain_model_acc8_path, knee_model_acc4_path, knee_model_acc8_path, classifier_path, device):
        brain_ckpt = torch.load(brain_model_acc4_path, map_location=device, weights_only=False)
        self.brain_net_acc4.load_state_dict(brain_ckpt['model'])

        brain_ckpt = torch.load(brain_model_acc8_path, map_location=device, weights_only=False)
        self.brain_net_acc8.load_state_dict(brain_ckpt['model'])

        knee_ckpt = torch.load(knee_model_acc4_path, map_location=device, weights_only=False)
        self.knee_net_acc4.load_state_dict(knee_ckpt['model'])

        knee_ckpt = torch.load(knee_model_acc8_path, map_location=device, weights_only=False)
        self.knee_net_acc8.load_state_dict(knee_ckpt['model'])

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
        acc = self.acc_classifier(masked_kspace)    # 4 or 8

        if choice.item() == 0:
            out = self.brain_net_acc4(masked_kspace, mask) if acc == 4 else self.brain_net_acc8(masked_kspace, mask)
        else:
            out = self.knee_net_acc4(masked_kspace, mask) if acc == 4 else self.knee_net_acc8(masked_kspace, mask)

        return out