"""MolGrapher keypoint detector (resnet18 + 2 conv head -> heatmap).

Faithful port of ``molgrapher/models/keypoint_detector.py`` with the
lightning / timm / mahotas dependencies removed.  Checkpoint
``models/keypoint_detector/kd_model.ckpt`` loads with zero missing/unexpected
keys (136 tensors).
"""
import torch
import torch.nn as nn

from .geometry import get_bond_size, regional_maxima
from .resnet import resnet18

__all__ = ["KeypointDetector", "MAX_KEYPOINTS"]

MAX_KEYPOINTS = 300


class KeypointDetector(nn.Module):
    """``feature_extractor`` (resnet18, dilation 8) + conv1/bn1/relu/conv2/bn2."""

    def __init__(self, nb_filters=512, dilation_factor=8, conv1_stride=2):
        super().__init__()
        self.feature_extractor = resnet18(output_layers=["layer4"],
                                          dilation_factor=dilation_factor,
                                          conv1_stride=conv1_stride)
        # Upstream derives this from ``list(children())[-3][-1].conv1.out_channels``
        # (=> 256).  The released kd_model.ckpt however has conv1 = 512 -> 256,
        # i.e. it was trained from the *output* width of layer4 (= 512).  The
        # value used here is the one the checkpoint proves; a strict
        # load_state_dict would fail otherwise.
        self.nb_filters = nb_filters
        self.conv1 = nn.Conv2d(self.nb_filters, self.nb_filters // 2, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.nb_filters // 2, eps=1e-05, momentum=0.1,
                                  affine=True, track_running_stats=True)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(self.nb_filters // 2, 1, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True,
                                  track_running_stats=True)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = self.feature_extractor(x)["layer4"]
        x = self.relu(self.bn1(self.conv1(x)))
        return self.bn2(self.conv2(x))

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(self, images, threshold=0.1, verbose=False):
        """images: (B, 3, 1024, 1024) -> {"keypoints_batch", "heatmaps_batch"}."""
        heatmaps = self.forward(images).cpu().detach()
        keypoints_batch = []
        for heatmap in heatmaps:
            heat = torch.where(heatmap > threshold, heatmap,
                               torch.zeros_like(heatmap))
            raw = heat.numpy()[0]
            # mahotas.regmax on the *thresholded* heatmap: the flat zero
            # background is one flat zone that touches higher values, hence it
            # is never a regional maximum -> keep only pixels above threshold.
            peaks = regional_maxima(raw, footprint=5) & (raw > threshold)
            keypoints_y, keypoints_x = peaks.nonzero()
            keypoints = [[int(x), int(y)] for x, y in zip(keypoints_x, keypoints_y)]

            if len(keypoints) > MAX_KEYPOINTS:
                print("Keypoints detection error: max number of keypoints exceeded")
                keypoints_batch.append(keypoints[:MAX_KEYPOINTS])
                continue
            if len(keypoints) == 0:
                print("Keypoints detection error: no keypoints detected")
                keypoints_batch.append(keypoints)
                continue
            if len(keypoints) == 1:
                print("Keypoints detection error: only one keypoint detected")
                keypoints_batch.append(keypoints)
                continue

            bond_size = get_bond_size(keypoints)
            if bond_size > 35:
                # For large molecules use a bigger footprint (upstream behaviour)
                peaks = regional_maxima(raw, footprint=15) & (raw > threshold)
                keypoints_y, keypoints_x = peaks.nonzero()
                keypoints = [[int(x), int(y)] for x, y in zip(keypoints_x, keypoints_y)]

            if verbose:
                print(f"keypoints={len(keypoints)} bond_size={bond_size:.2f}")
            keypoints_batch.append(keypoints)

        return {"keypoints_batch": keypoints_batch,
                "heatmaps_batch": [h for h in heatmaps]}
