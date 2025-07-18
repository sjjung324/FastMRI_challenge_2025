import numpy as np
import torch

from collections import defaultdict
from utils.common.utils import save_reconstructions
from utils.data.load_data import create_data_loaders
# from utils.model.varnet import VarNet
from utils.model.feature_varnet import FeatureVarNet_sh_w as VarNet
from utils.model.simple_classifier import SimpleClassifier

def test(args, brain_model, knee_model, classifier_model, data_loader):
    brain_model.eval()
    knee_model.eval()
    classifier_model.eval()
    reconstructions = defaultdict(dict)

    with torch.no_grad():
        for (mask, kspace, input_img, _, _, fnames, slices) in data_loader:
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            input_img = input_img.cuda(non_blocking=True)
            outputs = []
            # input_img가 3D (B, H, W) 또는 4D (B, C, H, W)인지 확인 후 classifier에 맞게 reshape
            if input_img.dim() == 3:
                input_img_batch = input_img.unsqueeze(1)  # (B, 1, H, W)
            else:
                input_img_batch = input_img  # (B, C, H, W)
            pred = classifier_model(input_img_batch)
            pred_label = torch.argmax(pred, dim=1)  # (B,)
            for i in range(kspace.shape[0]):
                kspace_i = kspace[i:i+1]
                mask_i = mask[i:i+1]
                if pred_label[i].item() == 0:
                    out = brain_model(kspace_i, mask_i)
                else:
                    out = knee_model(kspace_i, mask_i)
                outputs.append(out[0].cpu().numpy())
                reconstructions[fnames[i]][int(slices[i])] = out[0].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    return reconstructions, None


def forward(args):

    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print ('Current cuda device ', torch.cuda.current_device())

    # brain, knee varnet 모델 로드
    brain_model = VarNet(num_cascades=args.cascade_brain, chans=args.chans_brain, sens_chans=args.sens_chans_brain)
    knee_model = VarNet(num_cascades=args.cascade_knee, chans=args.chans_knee, sens_chans=args.sens_chans_knee)
    brain_ckpt = torch.load(args.brain_varnet_path, map_location=device, weights_only=False)
    knee_ckpt = torch.load(args.knee_varnet_path, map_location=device, weights_only=False)
    if 'model' in brain_ckpt:
        brain_model.load_state_dict(brain_ckpt['model'])
    else:
        brain_model.load_state_dict(brain_ckpt)
    if 'model' in knee_ckpt:
        knee_model.load_state_dict(knee_ckpt['model'])
    else:
        knee_model.load_state_dict(knee_ckpt)
    brain_model.to(device=device)
    knee_model.to(device=device)

    # classifier 모델 로드
    classifier_model = SimpleClassifier()
    classifier_ckpt = torch.load(args.classifier_path, map_location=device)
    if 'model' in classifier_ckpt:
        classifier_model.load_state_dict(classifier_ckpt['model'])
    else:
        classifier_model.load_state_dict(classifier_ckpt)
    classifier_model.to(device=device)
    classifier_model.eval()

    forward_loader = create_data_loaders(data_path = args.data_path, args = args, isforward = True, modality=args.modality)
    reconstructions, inputs = test(args, brain_model, knee_model, classifier_model, forward_loader)
    save_reconstructions(reconstructions, args.forward_dir, inputs=inputs)