#code for transforming from pth to onnx
import torch
import os
#used timm to construct model
import timm

#original file pth
pth_input = "predictor.pth"
#new onnx
onnx_output = "predictor.onnx"
#number of labels in dataset
class_count = 41
#each image is 224x224
dimension_size = 224

#constructs a basic model
def build_model(num_classes: int):
    #constructs the model that I used (vit tiny), sets its parameters by new one with number of classes
    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=False,
        num_classes=num_classes,   # <-- FIXED to use the function arg
    )
    return model
    
#loads the model from the checkpoint
def load_checkpoint(path: str, model: torch.nn.Module):
    #takes the checkpoint and loads it through torch to be able to adjust accordingly
    ckpt = torch.load(path, map_location="cpu")
    #assistance with difference checkpoint types used for saving model
    if isinstance(ckpt, torch.nn.Module):
        print("[INFO] Loaded full model from checkpoint (torch.save(model, ...)).")
        return ckpt
    #if the checkpoint is a dictionary instead of a model
    if isinstance(ckpt, dict):
        #check for the models state in model_state_dict and state_dict as precautionary measures
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            print("[INFO] Found 'model_state_dict' in checkpoint.")
        elif "state_dict" in ckpt:   # <-- FIXED syntax
            state_dict = ckpt["state_dict"]
            print("[INFO] Found 'state_dict' in checkpoint.")
        else:
            raise RuntimeError("Checkpoint missing model weights")
        #loads the state from dictionary to model and returns
        model.load_state_dict(state_dict)
        #returns loaded model
        return model
    #raise error if any flags
    raise RuntimeError("Checkpoint Issue with Model")

def main():
    #checks to see if checkpoint exists, if not raises error
    if not os.path.isfile(pth_input):   # <-- FIXED
        raise FileNotFoundError(f"Checkpoint not found: {pth_input}")

    #builds model
    model = build_model(class_count)   # <-- FIXED

    #loads weights onto the model
    model = load_checkpoint(pth_input, model)  # <-- FIXED
    model.eval()

    #onnx needs dummy input to adapt and store information for one forward pass
    dummy_input = torch.randn(1, 3, dimension_size, dimension_size, device="cpu")

    #exports to onnx, uses the model as base, saves name, gives example with a dummy input, etc.
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
    )
    #logs completion of export
    print("Export Complete")

if __name__ == "__main__":
    main()
