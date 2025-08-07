python train.py \
  -b 1 \
  -e 40 \
  --num-aug-epochs 20 \
  --using-noise-mask \
  -l 0.0002 \
  --alpha 0.84 \
  -r 10 \
  --cascade 20 \
  --chans 18 \
  --sens_chans 8 \
  --pools 4 \
  --sens_pools 4 \
  -m 'knee' \
  -n 'FeatureVarNet_20_18_8_knee_0807' \
  -t '../Data/train/' \
  -v '../Data/val/'
  #--pretrained-model-path '../result/FeatureVarNet_8_16_6_knee_aug_0721/checkpoints/best_model.pt' \