import os
import torch
import timm

pth_input = "predictor.pth"
onnx_output = "predictor12.onnx"
class_count = 41
dimension_size = 224


def build_model(num_classes: int):
    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=False,
        num_classes=num_classes,
    )
    return model


def load_checkpoint(path: str, model: torch.nn.Module):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, torch.nn.Module):
        print("[INFO] Loaded full model from checkpoint (torch.save(model, ...)).")
        return ckpt
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            print("[INFO] Found 'model_state_dict' in checkpoint.")
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            print("[INFO] Found 'state_dict' in checkpoint.")
        else:
            raise RuntimeError("Checkpoint missing model weights")
        model.load_state_dict(state_dict)
        return model
    raise RuntimeError("Checkpoint Issue with Model")


def main():
    if not os.path.isfile(pth_input):
        raise FileNotFoundError(f"Checkpoint not found: {pth_input}")

    model = build_model(class_count)
    model = load_checkpoint(pth_input, model)
    model.eval()

    dummy_input = torch.randn(1, 3, dimension_size, dimension_size, device="cpu")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_output,            # <- predictor12.onnx
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        use_external_data_format=False,  # <- single-file ONNX, no .data
    )

    print(f"Export complete: {onnx_output}")


if __name__ == "__main__":
    main()
