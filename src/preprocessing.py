#necessary inputs for preprocessing the data
import os
from typing import Dict, Any, List, Tuple, Union
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

#configuration of indices used
indices: List[int] = [
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

#only allows the labels in the indices list
ALLOWED_LABELS = set(indices)
#transfers indice labels to true indices 0-40
LABEL2IDX: Dict[int, int] = {lab: i for i, lab in enumerate(indices)}
#some files did get corrupted during transfer, so we just skip over corrupted files
corrupted = {
    "train": 0,
    "test": 0,
}


#parses label from file_name
def _parse_label_from_filename(filename: str) -> int:
    #takes filename
    base = os.path.basename(filename)
    #splits it
    stem, _ = os.path.splitext(base)
    #takes the first chunk which is the label
    first_chunk = stem.split("_")[0]
    #turns it into an integer value for actual usage
    label_int = int(first_chunk)
    #if the label is not valid we raise error
    if label_int not in ALLOWED_LABELS:
        raise ValueError(f"Label {label_int} from file {filename} not in ALLOWED_LABELS")
    #return the label 
    return label_int


def _list_image_files_with_labels(folder: str) -> Tuple[List[str], List[int]]:
    # only accept .jpg files (lowercased later)
    exts = {".jpg"}

    # accumulators for absolute paths and their corresponding label indices
    img_paths: List[str] = []
    label_idxs: List[int] = []
    
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)

        # skip directories / non-files
        if not os.path.isfile(fpath):
            continue

        # skip unsupported extensions
        _, ext = os.path.splitext(fname)
        if ext.lower() not in exts:
            continue

        # extract label string from filename and map to integer index
        label = _parse_label_from_filename(fname)
        label_idx = LABEL2IDX[label]

        img_paths.append(fpath)
        label_idxs.append(label_idx)

    # enforce deterministic ordering by sorting on file path
    paired = sorted(zip(img_paths, label_idxs), key=lambda x: x[0])
    img_paths, label_idxs = zip(*paired) if paired else ([], [])
    
    # return lists (zip gives tuples)
    return list(img_paths), list(label_idxs)

# grabs valid images (filters out corrupted/unreadable files)
def _filter_valid_images(
    image_paths: List[str],
    labels: List[int],
    split: str,
) -> Tuple[List[str], List[int]]:
    
    # enforce valid split (used to index corrupted counter)
    assert split in ("train", "test"), "split must be 'train' or 'test'"

    # accumulators for only intact, readable files
    valid_paths: List[str] = []
    valid_labels: List[int] = []

    for path, lab in zip(image_paths, labels):
        try:
            # lightweight integrity check; does not decode pixel data
            with Image.open(path) as img:
                img.verify()
        except Exception:
            # track and skip corrupted files for this split
            corrupted[split] += 1
            print(f"[WARN] Skipping corrupted {split} image: {path}")
            continue

        # file passed integrity check → include in dataset
        valid_paths.append(path)
        valid_labels.append(lab)

    # return filtered lists
    return valid_paths, valid_labels



# K DATASET CLASS, K={1,2,3,5}
class KImageDataset(Dataset):

    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        # ensure 1:1 mapping between image paths and label indices
        assert len(image_paths) == len(labels), "Mismatch between images and labels"
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        # dataset size = number of valid image paths
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        # fetch image file and its integer label
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # load image and enforce RGB channel format
        img = Image.open(img_path).convert("RGB")

        # apply user-specified transform pipeline if provided
        if self.transform is not None:
            img = self.transform(img)
        else:
            # fallback: only convert to tensor
            img = T.ToTensor()(img)

        # return tensor + label index
        return img, label
#basic transformation
def _get_base_transform() -> T.Compose:
    #mean and std based on imagenet
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    #resizes and normalizes
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

#returns distorted training, rather than basic training
def _get_distort_train_transform() -> T.Compose:
    #imagenet variables for normalization
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    #returns distorted images
    return T.Compose([
        #resizes image
        T.Resize((224, 224)),
        #adds randomness to color channels
        T.ColorJitter(
            brightness=0.2,    # ±20% brightness variation
            contrast=0.2,      # ±20% contrast variation
            saturation=0.2,    # ±20% saturation variation
            hue=0.1,
        ),
        #rotates image 15 degrees
        T.RandomRotation(degrees=15),  # rotate ±15 degrees
        #50 percent chance of another flip, which in turn leads to 25 percent prob of 15 degree, -15, 15 deg flip, -15 deg flip
        T.RandomHorizontalFlip(p=0.5),  # 50% chance of horizontal flip
        #tensorizes
        T.ToTensor(),
        #normalizes
        T.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

#the test shouldn't be distorted for testing, to see if bad data can generalize onto what would actually be tested in practice
def _get_distort_test_transform() -> T.Compose:
    return _get_base_transform()

#actually processes the images
def preprocess(
    k: Union[int, str],
    prep_type: str = "base",
    data_root: str = None,
) -> Dict[str, Any]:

    #takes k, and integerizes it
    if isinstance(k, str):
        k_str = k.lower()
        if k_str.startswith("k"):
            k_int = int(k_str[1:])
        else:
            k_int = int(k_str)
    else:
        k_int = int(k)
    #if k isn't one of these five values, I do not have a dataset for it
    if k_int not in {1, 2, 3, 5}:
        raise ValueError(f"k must be one of {{1,2,3,5}}, got {k_int}")

    #finds data root to collect from
    if data_root is None:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        data_root = os.path.join(this_dir, "data")

    #builds k-folder names and paths
    k_name = f"k{k_int}"
    k_dir = os.path.join(data_root, k_name)

    #train and test directories for this k-split
    train_dir = os.path.join(k_dir, f"{k_name}_train")
    test_dir = os.path.join(k_dir, f"{k_name}_test")

    #ensures folder structure exists
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    #lists all image files + mapped label indices
    train_paths, train_labels = _list_image_files_with_labels(train_dir)
    test_paths, test_labels = _list_image_files_with_labels(test_dir)

    # filters corrupted images
    train_paths, train_labels = _filter_valid_images(train_paths, train_labels, split="train")
    test_paths, test_labels = _filter_valid_images(test_paths, test_labels, split="test")

    # chooses which preparation type to use
    if prep_type == "base":
        train_transform = _get_base_transform()
        test_transform = _get_base_transform()
    elif prep_type == "distort":
        train_transform = _get_distort_train_transform()
        test_transform = _get_distort_test_transform()
    else:
        raise ValueError(
            f"Unknown prep_type '{prep_type}'. "
            f"Supported values: 'base', 'distort'"
        )

    #collects the trian and test dataset, applying necessary transforms
    train_dataset = KImageDataset(train_paths, train_labels, transform=train_transform)
    test_dataset = KImageDataset(test_paths, test_labels, transform=test_transform)

    #builds reverse index → label map
    idx2label = {idx: lab for lab, idx in LABEL2IDX.items()}

    #bundle being passed, e.g., the preparation type, the datasets, and labels/indices
    bundle = {
        "train": train_dataset,
        "test": test_dataset,
        "labels": indices,
        "label2idx": LABEL2IDX,
        "idx2label": idx2label,
        "k": k_int,
        "prep_type": prep_type,
    }

    return bundle

#basic sanity test you can try running directly to see if everything works appropraitly
if __name__ == "__main__":
    #some tests to run
    tests = [
        (1, "base"),
        (2, "base"),
        (3, "base"),
        (5, "base"),
        (1, "distort"),
        (2, "distort"),
        (3, "distort"),
        (5, "distort"),
    ]
    #iterates through each test
    for k_val, prep in tests:
        print(f"Testing k = {k_val}, prep_type = '{prep}'")

        #reset corrupted counters since different datasets have different corrupt count
        for key in corrupted:
            corrupted[key] = 0

        bundle = preprocess(k=k_val, prep_type=prep)

        train_ds = bundle["train"]
        test_ds = bundle["test"]

        print(f"num of train = {len(train_ds)}, num of test = {len(test_ds)}")

        if len(train_ds) > 0:
            x, y = train_ds[0]
            print("Sample X shape:", tuple(x.shape))
            print("Label idx:", y, "Label ID:", bundle["idx2label"][y])
        else:
            print("Train dataset is empty for this configuration.")

        print("Corrupted files skipped this run:")
        print("train:", corrupted["train"])
        print("test :", corrupted["test"])