import argparse
import torch
import torchvision
import torchvision.transforms.functional as TF

from src.utils import mask2rgb


def preprocess(image: torch.Tensor) -> torch.Tensor:
    image_0 = TF.resize(image, [512, 512])
    image_1 = torch.rot90(image_0, k=1, dims=[1, 2])
    image_2 = torch.rot90(image_0, k=2, dims=[1, 2])
    image_3 = torch.rot90(image_0, k=3, dims=[1, 2])
    image_4 = TF.hflip(image_0)
    image_5 = TF.vflip(image_0)

    batch = torch.stack([
            image_0, image_1, image_2,
            image_3, image_4, image_5,
        ], dim=0)

    return batch


def postprocess(mask: torch.Tensor) -> torch.Tensor:
    mask_0 = mask[0]
    mask_1 = torch.rot90(mask[1], k=3, dims=[1, 2])
    mask_2 = torch.rot90(mask[2], k=2, dims=[1, 2])
    mask_3 = torch.rot90(mask[3], k=1, dims=[1, 2])
    mask_4 = TF.hflip(mask[4])
    mask_5 = TF.vflip(mask[5])

    result = sum([mask_0, mask_1, mask_2,
                  mask_3, mask_4, mask_5])

    result = torch.argmax(result, dim=0)
    return mask2rgb(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='path to traced model')
    parser.add_argument('input', type=str, help='path to input file')
    parser.add_argument('--out', type=str, default='output.png', help='path to output image')
    args = parser.parse_args()

    model = torch.jit.load(args.model)
    image = torchvision.io.read_image(args.input)

    model_inp = preprocess(image)
    model_out = model(model_inp)
    model_out = postprocess(model_out)

    torchvision.utils.save_image(model_out / 255, args.out)
