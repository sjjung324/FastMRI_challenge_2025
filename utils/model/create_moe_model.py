import torch
import argparse
from pathlib import Path

from mixture_of_expert import MoEModel
    
def parse():
    parser = argparse.ArgumentParser(description='Create MoE model for FastMRI challenge', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--net_name', type=Path, default='test_varnet', help='Name of network')
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')

    # Brain_acc4 Varnet parameters
    parser.add_argument('--cascade_brain_acc4', type=int, default=1, help='Number of cascades | Should be less than 12')
    parser.add_argument('--chans_brain_acc4', type=int, default=9, help='Number of channels for cascade U-Net')
    parser.add_argument('--sens_chans_brain_acc4', type=int, default=4, help='Number of channels for sensitivity map U-Net')
    parser.add_argument('--pools_brain_acc4', type=int, default=4, help='Number of pooling layers for cascade U-Net | 4 in original varnet')
    parser.add_argument('--sens_pools_brain_acc4', type=int, default=4, help='Number of pooling layers for sensitivity map U-Net | 4 in original varnet')

    # Brain_acc8 Varnet parameters
    parser.add_argument('--cascade_brain_acc8', type=int, default=1, help='Number of cascades | Should be less than 12')
    parser.add_argument('--chans_brain_acc8', type=int, default=9, help='Number of channels for cascade U-Net')
    parser.add_argument('--sens_chans_brain_acc8', type=int, default=4, help='Number of channels for sensitivity map U-Net')
    parser.add_argument('--pools_brain_acc8', type=int, default=4, help='Number of pooling layers for cascade U-Net | 4 in original varnet')
    parser.add_argument('--sens_pools_brain_acc8', type=int, default=4, help='Number of pooling layers for sensitivity map U-Net | 4 in original varnet')

    # Knee_acc4 Varnet parameters
    parser.add_argument('--cascade_knee_acc4', type=int, default=1, help='Number of cascades | Should be less than 12')
    parser.add_argument('--chans_knee_acc4', type=int, default=9, help='Number of channels for cascade U-Net')
    parser.add_argument('--sens_chans_knee_acc4', type=int, default=4, help='Number of channels for sensitivity map U-Net')
    parser.add_argument('--pools_knee_acc4', type=int, default=4, help='Number of pooling layers for cascade U-Net | 4 in original varnet')
    parser.add_argument('--sens_pools_knee_acc4', type=int, default=4, help='Number of pooling layers for sensitivity map U-Net | 4 in original varnet')

    # Knee_acc8 Varnet parameters
    parser.add_argument('--cascade_knee_acc8', type=int, default=1, help='Number of cascades | Should be less than 12')
    parser.add_argument('--chans_knee_acc8', type=int, default=9, help='Number of channels for cascade U-Net')
    parser.add_argument('--sens_chans_knee_acc8', type=int, default=4, help='Number of channels for sensitivity map U-Net')
    parser.add_argument('--pools_knee_acc8', type=int, default=4, help='Number of pooling layers for cascade U-Net | 4 in original varnet')
    parser.add_argument('--sens_pools_knee_acc8', type=int, default=4, help='Number of pooling layers for sensitivity map U-Net | 4 in original varnet')

    # classifier, brain varnet, knee varnet 경로 인자
    parser.add_argument('--classifier_path', type=Path, required=True, help='Path to classifier model (.pt)')
    parser.add_argument('--brain_varnet_acc4_path', type=Path, required=True, help='Path to brain Varnet model (.pt)')
    parser.add_argument('--brain_varnet_acc8_path', type=Path, required=True, help='Path to brain Varnet model (.pt)')
    parser.add_argument('--knee_varnet_acc4_path', type=Path, required=True, help='Path to knee Varnet model (.pt)')
    parser.add_argument('--knee_varnet_acc8_path', type=Path, required=True, help='Path to knee Varnet model (.pt)')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    # brain FeatureVarNet arguments
    brain_model_acc4_args = {
        'num_cascades': args.cascade_brain_acc4,
        'chans': args.chans_brain_acc4,
        'sens_chans': args.sens_chans_brain_acc4,
        'pools': args.pools_brain_acc4,
        'sens_pools': args.sens_pools_brain_acc4
    }

    brain_model_acc8_args = {
        'num_cascades': args.cascade_brain_acc8,
        'chans': args.chans_brain_acc8,
        'sens_chans': args.sens_chans_brain_acc8,
        'pools': args.pools_brain_acc8,
        'sens_pools': args.sens_pools_brain_acc8
    }

    # knee FeatureVarNet arguments
    knee_model_acc4_args = {
        'num_cascades': args.cascade_knee_acc4,
        'chans': args.chans_knee_acc4,
        'sens_chans': args.sens_chans_knee_acc4,
        'pools': args.pools_knee_acc4,
        'sens_pools': args.sens_pools_knee_acc4
    }

    knee_model_acc8_args = {
        'num_cascades': args.cascade_knee_acc8,
        'chans': args.chans_knee_acc8,
        'sens_chans': args.sens_chans_knee_acc8,
        'pools': args.pools_knee_acc8,
        'sens_pools': args.sens_pools_knee_acc8
    }

    # MoE 모델 생성
    moe_model = MoEModel(brain_model_acc4_args=brain_model_acc4_args, brain_model_acc8_args=brain_model_acc8_args, knee_model_acc4_args=knee_model_acc4_args, knee_model_acc8_args=knee_model_acc8_args)
    moe_model.load_models(
        brain_model_acc4_path=args.brain_varnet_acc4_path,
        brain_model_acc8_path=args.brain_varnet_acc8_path,
        knee_model_acc4_path=args.knee_varnet_acc4_path,
        knee_model_acc8_path=args.knee_varnet_acc8_path,
        classifier_path=args.classifier_path,
        device=device   
    )

    # 모델 저장
    moe_model_path = Path('../result') / args.net_name / 'best_model.pt'
    moe_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': moe_model.state_dict()}, moe_model_path)
