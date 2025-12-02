#necessary imports
import os
import random
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import onnxruntime as ort

#dummy script to pass
model_dir = "predictor.onnx"
image_dir = "../data/k1/k1_test"

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

#swaps labels to indices based on what I had
LABEL2IDX = {lab: i for i, lab in enumerate(indices)}
IDX2LABEL = {i: lab for lab, i in LABEL2IDX.items()}

#ids <-> names for easy human legibilitiy
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
    801: "Health Sciences I",
    802: "Health Sciences II",
}

#sets preprocessing based on image net, what the vit was pretrained on
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

#forces to resize to 224, tensorize, and normalize for efficient processing
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=imagenet_mean, std=imagenet_std),
])

#takes the first part of the title of the image and classify as the true label
def parse_label_from_filename(filename: str) -> int:
    #takes the filename
    base = os.path.basename(filename)
    #splits the filename
    stem, _ = os.path.splitext(base)
    #takes the first chunk as in we have image input forms labelid_videoid_frameid
    first_chunk = stem.split("_")[0]
    #returns true label
    label_int = int(first_chunk)
    return label_int
    
#picks a random image, 
def pick_random_image(img_dir: str):
    #we only care for jpg in my images, for generlization it would be best to transfer from other images to jpg to match my model
    exts = {".jpg"}
    #collects all cadindates in the directory to see if they are jpg.
    candidates = [os.path.join(img_dir, f)
                  for f in os.listdir(img_dir)
                  if os.path.splitext(f)[1].lower() in exts]
    
    #if none found raise error
    if not candidates:
        raise RuntimeError(f"No images found in {img_dir}")

    #chooses a random one and returns it
    path = random.choice(candidates)
    return path

#main
def main():
    #sets up the model/session
    print("Loading ONNX model")
    sess = ort.InferenceSession(
        model_dir,  # FIXED
        providers=["CPUExecutionProvider"],
    )
    #obtains inputs and outputs from model
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    
    #picks a random image
    img_path = pick_random_image(image_dir)  # FIXED
    #gets its out id
    true_label_id = parse_label_from_filename(img_path)

    #gets true label idx as it swaps from my hundreds calculator to #1-41
    true_label_idx = LABEL2IDX[true_label_id]
    true_name = buildings.get(true_label_id, f"Building {true_label_id}")  # FIXED

    print(f"Random test image: {img_path}")
    print(f"True label ID: {true_label_id} (idx {true_label_idx})")
    print(f"True building: {true_name}")

    #preprocesses the image, opens image in RGB, transforms image as necessary, and turns into vector for usage.
    img = Image.open(img_path).convert("RGB")
    x = transform(img)
    x = x.unsqueeze(0)
    x_np = x.numpy().astype(np.float32)

    #takes outputs by passing through model
    outputs = sess.run([output_name], {input_name: x_np})
    logits = outputs[0]
    logits = logits[0]

    #makes a prediction
    pred_idx = int(np.argmax(logits))
    pred_label_id = IDX2LABEL[pred_idx]
    pred_name = buildings.get(pred_label_id, f"Building {pred_label_id}")  # FIXED

    #prints results
    print("\nRESULTS")
    print(f"Predicted index: {pred_idx}")
    print(f"Predicted label ID: {pred_label_id}")
    print(f"Predicted building: {pred_name}")

    #shows the five highest logits picked and their probability distribution
    top5_idx = np.argsort(logits)[-5:][::-1]
    print("\nTop-5 classes (idx, label_id, building, logit):")
    for i in top5_idx:
        lab_id = IDX2LABEL[int(i)]
        name = buildings.get(lab_id, f"Building {lab_id}")  # FIXED
        print(f"  {int(i):2d} | {lab_id:3d} | {name} | {logits[i]:.3f}")


if __name__ == "__main__":
    main()