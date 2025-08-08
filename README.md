# GJG
2025 SNU FastMRI challenge Team GJG


## 1. 폴더 정보

```bash
├── .gitignore
├── leaderboard_eval.py
├── README.md
├── reconstruct.py
├── requirements.txt
├── train.py
├── train_classifier.py
├── scripts
│   ├── create_moe_model.sh
│   ├── leaderboard_eval.sh
│   ├── reconstruct.sh
│   ├── train_classifier.sh
│   └── train.sh
├── utils
│   ├── common
│   │   ├── loss_function.py
│   │   └── utils.py
│   ├── data
│   │   ├── augmentation.py
│   │   ├── load_data.py
│   │   ├── load_image_data.py
│   │   └── transforms.py
│   ├── learning
│   │   ├── fastmri
│   │   └── train_part.py
│   └── model
│       ├── create_moe_model.py
│       ├── feature_varnet.py
│       ├── mixture_of_expert.py
│       ├── not_baby_unet.py
│       ├── simple_classifier.py
│       ├── unet.py
│       └── varnet.py
└── result
```

## 2. Training 순서
- Train Classifier Model First (Brain/Knee 구분, MoE 모델에 이용)
```bash
sh scrpits/train_classifier.sh
```
- FeatureVarnet based brain model training
```bash
sh scrpits/train_brain.sh
```
- FeatureVarnet based knee model training
```bash
sh scrpits/train_knee.sh
```
- Combine pretrained three models to MoE model
```bash
sh scrpits/create_moe_model.sh
```
- Reconstruct using MoE model
```bash
sh scrpits/create_moe_model.sh
```