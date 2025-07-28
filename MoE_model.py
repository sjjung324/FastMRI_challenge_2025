import torch
import torch.nn as nn

class MoEModel(nn.Module):
    def __init__(self, brain_net, knee_net, classifier):
        super().__init__()
        self.brain_net = brain_net       # 서브모델
        self.knee_net  = knee_net
        self.classifier = classifier     # 예: brain vs knee 판단용

    def forward(self, x):
        # 분기 예시 — 원하는 로직으로 바꾸세요
        logits = self.classifier(x)
        choice = logits.argmax(dim=1)        # 0: brain, 1: knee (batch 단순 예)
        out = torch.empty_like(x)

        brain_mask = choice == 0
        knee_mask  = choice == 1

        if brain_mask.any():
            out[brain_mask] = self.brain_net(x[brain_mask])
        if knee_mask.any():
            out[knee_mask]  = self.knee_net(x[knee_mask])

        return out

# ① 각각의 네트워크 인스턴스 생성 + 파라미터 로드
brain_net = BrainNet(); brain_net.load_state_dict(torch.load("brain.pt"))
knee_net  = KneeNet();  knee_net.load_state_dict(torch.load("knee.pt"))
cls_net   = Classifier(); cls_net.load_state_dict(torch.load("classifier.pt"))

# ② 래퍼 모델 생성
moe_model = MoEModel(brain_net, knee_net, cls_net)

# ③ 통째로 저장 (아키텍처 + 파라미터)
torch.save(moe_model, "moe_model.pt")
