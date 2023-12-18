import pickle
import cv2
import numpy as np
import gzip
import os
from tqdm import tqdm
import json

def illustrate_ith_frame_of_ith_video(data, i, pick_annotated=False):
    print(data[i]["dataset"])
    
    annotated_frames_idxs = data[i]["frames"]
    frame_idx = np.random.choice(annotated_frames_idxs) if pick_annotated else np.random.choice(range(data[i]["video"].shape[2]))
    frame = data[i]["video"][:,:,frame_idx]
    labels = data[i]["label"][:,:,frame_idx]
    box = data[i]["box"]
    
    bbox_coords = np.where(box)
    bbox = np.array([np.min(bbox_coords[1]), np.min(bbox_coords[0]), np.max(bbox_coords[1]), np.max(bbox_coords[0])])
    
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,0,0), 2)
    frame[labels==1] = [0,255,0]
    
    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
def binary_mask_to_coco_format(binary_mask):
    # Ensure the input mask is a NumPy array
    mask_array = np.asarray(binary_mask, dtype=np.uint8)
    x, y, w, h = cv2.boundingRect(mask_array)

    # Find contours in the binary mask
    # add third dimension to mask_array
    contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Convert the contours to COCO format
    coco_format = []
    for contour in contours:
        # Each contour is represented as a list of [x, y] coordinates
        contour_coordinates = contour.flatten().tolist()
        coco_format.append(contour_coordinates)

    return coco_format, [x, y, w, h]


def save_annotated_expert_frames_as_coco(data):
    """Save annotated frames of expert dataset as COCO dataset."""
    dts_dir = 'datasets/coco'
    anns = {"train": {"images": [], "annotations": [], "categories": [{"id": 1, "name": "mvalave"}]}, 
            "val": {"images": [], "annotations": [], "categories": [{"id": 1, "name": "mvalave"}]}}
    
    
    for i in tqdm(range(len(data))):
        if data[i]["dataset"] == "expert":
            annotated_frames_idxs = data[i]["frames"]
            
            for frame_idx in annotated_frames_idxs:
                split = "val" if len(anns["val"]["images"]) < 1 else "train"
                frame = data[i]["video"][:,:,frame_idx]
                mask = data[i]["label"][:,:,frame_idx]
                #box = data[i]["box"]
                anns[split]["images"].append({"id": len(anns[split]["images"]), 
                                              "file_name": f'{len(anns[split]["images"])}.png', 
                                              "width": frame.shape[1], 
                                              "height": frame.shape[0]})
                cv2.imwrite(f'{dts_dir}/{split}2017/{len(anns[split]["images"])-1}.png', frame)
                maskcoco, bbox = binary_mask_to_coco_format(mask)
                anns[split]["annotations"].append({"id": len(anns[split]["annotations"]),
                                                    "image_id": len(anns[split]["images"])-1,
                                                    "category_id": 1,
                                                    "segmentation": maskcoco,
                                                    "area": int(np.sum(mask)),
                                                    "bbox": bbox,
                                                    "iscrowd": 0})
                
    for split in ["train", "val"]:
        with open(f'{dts_dir}/annotations/instances_{split}2017.json', 'w') as f:
            json.dump(anns[split], f)


def save_all_imgs(data):
    datadir = 'dataset'
    val_split = 0.2 * len(data)

    for i in tqdm(range(len(data))):
        split = "val" if i < val_split else "train"
        for j in range(data[i]["video"].shape[2]):
            frame = data[i]["video"][:,:,j]
            cv2.imwrite(f'{datadir}/{split}/{i}_{j}.png', frame)  


def save_all_masks(data):
    datadir = 'dataset'

    for i in tqdm(range(len(data))):
        annotated_frames_idxs = data[i]["frames"]
        for j in annotated_frames_idxs:
            mask = data[i]["label"][:,:,j]
            if os.path.exists(f'{datadir}/train/{i}_{j}.png'):
                split = "train_gt"
            elif os.path.exists(f'{datadir}/val/{i}_{j}.png'):
                split = "val_gt"
            else:
                raise Exception("Image not found")
            cv2.imwrite(f'{datadir}/{split}/{i}_{j}.png', (mask*255).astype(np.uint8))                                                          
                                                
    
def save_all_inpainting_imgs(data):
    datadir = 'dataset'
    val_split = 0.2 * len(data)

    for i in tqdm(range(len(data))):
        split = "val" if i < val_split else "train"
        box = data[i]["box"]
        for j in range(data[i]["video"].shape[2]):
            frame = data[i]["video"][:,:,j]
            # zero all pixels inside the bounding box
            frame[box==1] = 0
            cv2.imwrite(f'{datadir}_inp/{split}/{i}_{j}.png', frame)
    
        
    

# Load zipped pickle file
with gzip.open('task3/train.pkl', 'rb') as f:
    data = pickle.load(f)


#idxs_of_dataset_eq_expert = [i for i in range(len(data)) if data[i]["dataset"] == "expert"]
#i = np.random.choice(idxs_of_dataset_eq_expert)

#illustrate_ith_frame_of_ith_video(data, i, pick_annotated=True)
#save_annotated_expert_frames_as_coco(data)

#save_all_imgs(data)
#save_all_masks(data)
save_all_inpainting_imgs(data)

"""print(data[i]["dataset"])
print(f'Video dimensions: {data[i]["video"].shape}')
print(f'Bboxes dimensions: {data[i]["box"].shape}')
print(f'Labels dimensions: {data[i]["label"].shape}')
print(data[i]["box"])

annotated_frames_idxs = data[i]["frames"]

print(f'Annotated frames: {annotated_frames_idxs}')
print(np.sum(data[i]["label"][:,:,0]))
print(np.sum(data[i]["label"][:,:,59]))"""

