"""
ViT backbone and downstream heads for BrainIAC.
Copied from BrainIAC_V2/src/model.py so mechinterp can run SAE training without PYTHONPATH to the full repo.
"""
import torch
import torch.nn as nn
from monai.networks.nets import ViT
import torch.nn.functional as F
import yaml


class ViTBackboneNet(nn.Module):
    def __init__(self, simclr_ckpt_path):
        super(ViTBackboneNet, self).__init__()

        # Create ViT backbone with same architecture as SimCLR
        self.backbone = ViT(
            in_channels=1,
            img_size=(96, 96, 96),
            patch_size=(16, 16, 16),
            hidden_size=768,
            mlp_dim=3072,
            num_layers=12,
            num_heads=12,
            save_attn=True,
        )

        # Load pretrained weights from SimCLR checkpoint
        ckpt = torch.load(simclr_ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)

        backbone_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("backbone."):
                new_key = key[9:]
                backbone_state_dict[new_key] = value

        # strict=False: newer MONAI ViT has norm_cross_attn/cross_attn per block (unused when
        # with_cross_attention=False); BrainIAC checkpoint was trained with older ViT without those keys.
        missing, unexpected = self.backbone.load_state_dict(backbone_state_dict, strict=False)
        if missing:
            print(f"  (Backbone: {len(missing)} keys not in checkpoint, e.g. cross_attn - left as init)")
        if unexpected:
            print(f"  (Checkpoint: {len(unexpected)} keys not loaded)")
        print("Backbone weights loaded!!")

    def forward(self, x):
        features = self.backbone(x)
        cls_token = features[0][:, 0]
        return cls_token


class Classifier(nn.Module):
    def __init__(self, d_model=768, num_classes=1):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x


class SingleScanModel(nn.Module):
    def __init__(self, backbone, classifier):
        super(SingleScanModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


class SingleScanModelBP(nn.Module):
    def __init__(self, backbone, classifier):
        super(SingleScanModelBP, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        scan_features_list = []
        for scan_tensor_with_extra_dim in x.split(1, dim=1):
            squeezed_scan_tensor = scan_tensor_with_extra_dim.squeeze(1)
            feature = self.backbone(squeezed_scan_tensor)
            scan_features_list.append(feature)
        stacked_features = torch.stack(scan_features_list, dim=1)
        merged_features = torch.mean(stacked_features, dim=1)
        merged_features = self.dropout(merged_features)
        output = self.classifier(merged_features)
        return output


class SingleScanModelQuad(nn.Module):
    def __init__(self, backbone, classifier):
        super(SingleScanModelQuad, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        image1, image2, image3, image4 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        features1 = self.backbone(image1)
        features2 = self.backbone(image2)
        features3 = self.backbone(image3)
        features4 = self.backbone(image4)
        stacked_features = torch.stack([features1, features2, features3, features4], dim=1)
        merged_features = torch.mean(stacked_features, dim=1)
        merged_features = self.dropout(merged_features)
        output = self.classifier(merged_features)
        return output
