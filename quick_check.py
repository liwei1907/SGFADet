"""Run a small SGFADet forward pass without datasets or MobileSAM weights."""

from __future__ import annotations

import argparse

import torch

from sgfadet import SGFADet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=128)
    args = parser.parse_args()
    if args.image_size < 64 or args.image_size % 32:
        raise ValueError("--image-size must be at least 64 and divisible by 32")

    device = torch.device(args.device)
    model = SGFADet(variant="full").to(device).eval()
    image = torch.randn(1, 3, args.image_size, args.image_size, device=device)
    prior = torch.randn(1, 256, 64, 64, device=device)
    with torch.inference_mode():
        output = model(image, prior)

    expected = (1, 1, args.image_size, args.image_size)
    assert tuple(output["logits"].shape) == expected
    assert tuple(output["semantic"].shape) == expected
    assert tuple(output["boundary"].shape) == expected
    assert len(output["sides"]) == 4
    assert all(tuple(side.shape) == expected for side in output["sides"])
    print({key: [tuple(v.shape) for v in value] if isinstance(value, list) else tuple(value.shape)
           for key, value in output.items()})
    print(f"parameters={sum(parameter.numel() for parameter in model.parameters()):,}")


if __name__ == "__main__":
    main()
