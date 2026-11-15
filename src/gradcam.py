"""Grad-CAM (Selvaraju i in., 2017) - własna prosta implementacja na hookach.

Mapa = ReLU( suma po kanałach: waga_kanału * aktywacja_kanału ), gdzie waga kanału
to średni gradient wyniku klasy po aktywacjach ostatniej warstwy konwolucyjnej.
"""
import numpy as np
import torch
import torch.nn.functional as F


def grad_cam(model, x, target=None):
    """x: tensor 1 x 1 x H x W (wartości 0-1). Zwraca (mapa H x W w [0, 1], klasa, prawdopodobieństwa)."""
    model.eval()
    layer = model.cam_layer()
    store = {}

    def save(module, inputs, output):
        # zapamiętujemy aktywacje i rejestrujemy hook na ich gradient
        store['act'] = output
        output.register_hook(lambda g: store.__setitem__('grad', g))

    handle = layer.register_forward_hook(save)
    try:
        x = x.clone().requires_grad_(True)  # potrzebne, gdy pierwsze warstwy są zamrożone
        logits = model(x)
        probs = torch.softmax(logits, 1)[0].detach().cpu().numpy()
        if target is None:
            target = int(probs.argmax())
        model.zero_grad()
        logits[0, target].backward()
    finally:
        handle.remove()
    act, grad = store['act'][0], store['grad'][0]          # C x h x w
    weights = grad.mean(dim=(1, 2))                         # C
    cam = F.relu((weights[:, None, None] * act).sum(0))     # h x w
    cam = F.interpolate(cam[None, None], size=x.shape[-2:], mode='bilinear', align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam, target, probs


def overlay(image, cam, alpha=0.4):
    """Nakłada mapę (kolormapa jet) na obraz szary; zwraca RGB uint8."""
    import matplotlib.cm as cm
    gray = np.repeat(np.asarray(image, dtype=np.float32)[..., None] / 255.0, 3, axis=2)
    heat = cm.jet(cam)[..., :3]
    return (255 * ((1 - alpha) * gray + alpha * heat)).clip(0, 255).astype(np.uint8)
