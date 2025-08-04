import torch
import argparse
from pathlib import Path

from mixture_of_expert import MoEModel
    
def parse():
    parser = argparse.ArgumentParser(description='Create MoE model for FastMRI challenge',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--net_name', type=Path, default='test_varnet', help='Name of network')
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')

    # Brain Varnet parameters
    parser.add_argument('--cascade_brain', type=int, default=1, help='Number of cascades | Should be less than 12')
    parser.add_argument('--chans_brain', type=int, default=9, help='Number of channels for cascade U-Net')
    parser.add_argument('--sens_chans_brain', type=int, default=4, help='Number of channels for sensitivity map U-Net')
    parser.add_argument('--pools_brain', type=int, default=4, help='Number of pooling layers for cascade U-Net | 4 in original varnet')
    parser.add_argument('--sens_pools_brain', type=int, default=4, help='Number of pooling layers for sensitivity map U-Net | 4 in original varnet')

    # Knee Varnet parameters
    parser.add_argument('--cascade_knee', type=int, default=1, help='Number of cascades | Should be less than 12')
    parser.add_argument('--chans_knee', type=int, default=9, help='Number of channels for cascade U-Net')
    parser.add_argument('--sens_chans_knee', type=int, default=4, help='Number of channels for sensitivity map U-Net')
    parser.add_argument('--pools_knee', type=int, default=4, help='Number of pooling layers for cascade U-Net | 4 in original varnet')
    parser.add_argument('--sens_pools_knee', type=int, default=4, help='Number of pooling layers for sensitivity map U-Net | 4 in original varnet')

    # classifier, brain varnet, knee varnet 경로 인자
    parser.add_argument('--classifier_path', type=Path, required=True, help='Path to classifier model (.pt)')
    parser.add_argument('--brain_varnet_path', type=Path, required=True, help='Path to brain Varnet model (.pt)')
    parser.add_argument('--knee_varnet_path', type=Path, required=True, help='Path to knee Varnet model (.pt)')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

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

    # MoE 모델 생성
    moe_model = MoEModel(brain_model_args=brain_model_args, knee_model_args=knee_model_args)
    moe_model.load_models(
        brain_model_path=args.brain_varnet_path,
        knee_model_path=args.knee_varnet_path,
        classifier_path=args.classifier_path,
        device=device
    )

    # 모델 저장
    moe_model_path = Path('../result') / args.net_name / 'best_model.pt'
    moe_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': moe_model.state_dict()}, moe_model_path)
