"""Grad-CAM: which pixels pushed the model towards its prediction? (Selvaraju et al., 2017)

Weights each channel of the last conv layer by the average gradient of the class score, sums them,
keeps the positive part and upsamples to the image size.
"""
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, layer):
        self.model = model.eval()
        self.acts = self.grads = None
        layer.register_forward_hook(lambda m, i, o: setattr(self, "acts", o))
        layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "grads", go[0]))

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> tuple:
        """x: one image (1, 3, H, W). Returns (heatmap HxW in [0, 1], predicted class, probability)."""
        self.model.zero_grad()
        logits = self.model(x)
        probs = logits.softmax(1)[0]
        cls = int(probs.argmax()) if class_idx is None else class_idx
        logits[0, cls].backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.detach().cpu().numpy(), cls, probs[cls].item()
