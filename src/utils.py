import torch

_rgb2roomtype = {
        0: [255,255,255], # background
        1: [192,192,224], # closet
        2: [192,255,255], # bathroom
        3: [224,255,192], # livingroom/kitchen
        4: [255,224,128], # bedroom
        5: [255,160, 96], # hall
        6: [255,224,224], # balcony
        7: [224,224,224], # unused
        8: [224,224,128], # unused,
        9: [  0,  0,  0], # wall
       10: [255,  0,  0], # windows & doors
    }

def mask2rgb(mask: torch.Tensor) -> torch.Tensor:
    height, width = mask.shape
    image = torch.zeros(height, width, 3, dtype=torch.uint8)

    for class_idx, rgb in _rgb2roomtype.items():
        rgb = torch.tensor(rgb, dtype=torch.uint8)
        image[mask == class_idx] = rgb

    image = image.permute(2, 0, 1)
    return image
