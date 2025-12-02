#necessary imports
import os
import sys
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import onnxruntime as ort

#path to the onnx model
model_dir = "predictor.onnx"

#all label ids for classification in my program
indices = [
    101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
    111, 112, 113,
    201, 202, 203, 204,
    301, 302, 303,
    401, 402, 403, 404, 405,
    501,
    601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612,
    701,
    801, 802,
]

#mapping for idx <-> label
LABEL2IDX = {lab: i for i, lab in enumerate(indices)}
IDX2LABEL = {i: lab for lab, i in LABEL2IDX.items()}

#names for buildings based on id
buildings = {
    101: "Classroom Building 1",
    102: "Classroom Building 2",
    103: "College of Arts and Humanities",
    104: "Education Complex",
    105: "Howard Phillips Hall",
    106: "Math and Sciences Building",
    107: "Nicholson School of Communication and Media",
    108: "Teaching Academy",
    109: "Trevor Colbourn Hall",
    110: "Business Administration Buildings",
    111: "Counseling and Psychology Services",
    112: "College of Sciences Building",
    113: "Burnett Honors College",
    201: "Biological Sciences",
    202: "Chemistry Building",
    203: "Physical Sciences",
    204: "Psychology Building",
    301: "Engineering Buildings",
    302: "L3Harris Engineering Center",
    303: "CREOL – College of Optics & Photonics",
    401: "Performing Arts – Music",
    402: "Performing Arts – Theatre",
    403: "Theatre",
    404: "Rehearsal Hall",
    405: "Visual Arts Building",
    501: "John C. Hitt Library",
    601: "Student Union",
    602: "John T. Washington Center",
    603: "63 South",
    604: "Tech Commons Buildings",
    605: "Health Center",
    606: "General Ferrell Commons",
    607: "Live Oak Event Center (Live Oak Ballroom)",
    608: "Knights Pantry",
    609: "Research 1",
    610: "Career Services and Experiential Learning",
    611: "FAIRWINDS Alumni Center",
    612: "UCF Global",
    701: "Millican Hall",
    801: "Health Sciences 1",
    802: "Health Sciences 2",
}

#preprocessing based on imagenet stats used by vit tiny
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

#forces to resize to 224x224, convert to tensor, normalize
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=imagenet_mean, std=imagenet_std),
])

#convert png/webp/heic/etc to jpg for consistent model input
def convert_to_jpg(path: str) -> str:
    _, ext = os.path.splitext(path)
    ext = ext.lower()

    #jpg already correct input
    if ext in [".jpg", ".jpeg"]:
        return path

    #convert image to rgb then export as jpg
    img = Image.open(path).convert("RGB")
    new_path = path + ".jpg"
    img.save(new_path, "JPEG")
    return new_path

#main classifier that takes a direct file path
def main():
    #ensures user provided an argument
    if len(sys.argv) < 2:
        raise RuntimeError("Please call: python generalclassifier.py <image_path>")

    #user provided image path
    img_path = sys.argv[1]

    #checks file exists
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"File not found: {img_path}")

    #load ONNX session
    print(f"[INFO] Loading ONNX model from: {model_dir}")
    sess = ort.InferenceSession(
        model_dir,
        providers=["CPUExecutionProvider"],
    )

    #get model input & output names
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    #convert strange extension → jpg
    img_path_jpg = convert_to_jpg(img_path)
    print(f"[INFO] Using processed image: {img_path_jpg}")

    #opens image for preprocessing
    img = Image.open(img_path_jpg).convert("RGB")

    #apply 224x224 + normalize
    x = transform(img)
    x = x.unsqueeze(0)
    x_np = x.numpy().astype(np.float32)

    #run through model
    outputs = sess.run([output_name], {input_name: x_np})
    logits = outputs[0][0]

    #get predicted class index
    pred_idx = int(np.argmax(logits))
    pred_label_id = IDX2LABEL[pred_idx]
    pred_name = buildings.get(pred_label_id, f"Building {pred_label_id}")

    #print results just like your UCF-specific classifier
    print("\nRESULTS")
    print(f"Predicted index: {pred_idx}")
    print(f"Predicted label ID: {pred_label_id}")
    print(f"Predicted building: {pred_name}")

    #top 5 logits by descending order
    top5_idx = np.argsort(logits)[-5:][::-1]

    print("\nTop-5 classes (idx, label_id, building, logit):")
    for i in top5_idx:
        lab_id = IDX2LABEL[int(i)]
        name = buildings.get(lab_id, f"Building {lab_id}")
        print(f"  {int(i):2d} | {lab_id:3d} | {name} | {logits[i]:.3f}")


if __name__ == "__main__":
    main()