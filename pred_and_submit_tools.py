from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from maskdino import add_maskdino_config
from detectron2.projects.deeplab import add_deeplab_config

import numpy as np
from tqdm import tqdm
import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import os
from tqdm import tqdm
import gzip
import pickle
    

def setup_cfg(config_file):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.freeze()
    #print(f"* * *MODEL WEIGHTS: * * * {cfg.MODEL.WEIGHTS}")
    return cfg

def make_predictor(config_file):
    cfg = setup_cfg(config_file)     
    predictor = DefaultPredictor(cfg)
    return predictor, cfg

def show_val(parts_model):
    for img_path in os.listdir('datasets/coco/val2017'):
        img = cv2.imread('datasets/coco/val2017/' + img_path)
        
        outputs = parts_model(img)
        predictions = outputs['instances'][outputs['instances'].scores > 0.05]
        pred_masks = predictions.pred_masks.cpu().numpy()
        pred_masks = np.sum(pred_masks, axis=0) > 0
        
        #cv2.imshow('pred bin mask', pred_masks.astype(np.uint8)*255)
        #cv2.waitKey(0)
        
        v = Visualizer(img, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        v = v.draw_instance_predictions(predictions.to("cpu"))
        cv2.imshow('vd', v.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        
def do_submission(parts_model):
    with gzip.open('task3/test.pkl', 'rb') as f:
        data = pickle.load(f)
    
    predictions_f = []
    for d in tqdm(data):
        prediction_for_video = []
        for frame_idx in range(d['video'].shape[2]):
            frame = d['video'][:,:,frame_idx]
            #print(frame.shape)
            #cv2.imshow('frame', frame)
            #cv2.waitKey(0)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            outputs = parts_model(frame)
            predictions = outputs['instances'][outputs['instances'].scores > 0.05]
            pred_masks = predictions.pred_masks.cpu().numpy()
            pred_masks = np.sum(pred_masks, axis=0) > 0
            prediction = pred_masks.astype(np.uint8)
            prediction_for_video.append(prediction)
            #cv2.imshow('pred bin mask', prediction*255)
            #cv2.waitKey(0)
        
        predictions_f.append({
            'name': d['name'],
            'prediction': np.stack(prediction_for_video, axis=-1).astype(np.bool)
            }
        )
        
    with gzip.open('task3/submission_ft_on_big.pkl', 'wb') as f:
        pickle.dump(predictions_f, f, 2)
    


#config = 'output/aml_amadino_R50_bs16_50ep_3s_dowsample1_2048/config.yaml'
config = 'output/aml_ft_amadino_R50_bs16_50ep_3s_dowsample1_2048/config.yaml'
parts_model, cfg = make_predictor(config)

#show_val(parts_model)

do_submission(parts_model)

