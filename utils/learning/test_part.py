import numpy as np
import torch
from pathlib import Path

from collections import defaultdict
from utils.common.utils import save_reconstructions
from utils.data.load_data import create_data_loaders
from utils.model.mixture_of_expert import MoEModel

def test(args, moe_model, data_loader):
    moe_model.eval()
    reconstructions = defaultdict(dict)

    with torch.no_grad():
        for (mask, kspace, input_img, _, _, fnames, slices) in data_loader:
            kspace = kspace.cuda(non_blocking=True) # (S, C, H, W, 2)
            mask = mask.cuda(non_blocking=True)
            input_img = input_img.cuda(non_blocking=True)

            for i in range(kspace.shape[0]):
                out = moe_model(kspace[i:i+1], mask[i:i+1], input_img[i:i+1].unsqueeze(1))
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

    # MoE 모델 로드
    # brain FeatureVarNet arguments
    brain_model_args = {
        'num_cascades': args.cascade_brain,
        'chans': args.chans_brain,
        'sens_chans': args.sens_chans_brain,
        'pools': args.pools_brain,
        'sens_pools': args.sens_pools_brain
    }

    # knee FeatureVarNet arguments
    knee_model_args = {
        'num_cascades': args.cascade_knee,
        'chans': args.chans_knee,
        'sens_chans': args.sens_chans_knee,
        'pools': args.pools_knee,
        'sens_pools': args.sens_pools_knee
    }
    moe_model = MoEModel(brain_model_args=brain_model_args, knee_model_args=knee_model_args)
    moe_model_path = Path('../result') / args.net_name / 'best_model.pt'
    moe_ckpt = torch.load(moe_model_path, map_location=device, weights_only=False)
    moe_model.load_state_dict(moe_ckpt['model'])
    moe_model.to(device)

    forward_loader = create_data_loaders(data_path = args.data_path, args = args, isforward = True)
    reconstructions, inputs = test(args, moe_model, forward_loader)
    save_reconstructions(reconstructions, args.forward_dir, inputs=inputs)