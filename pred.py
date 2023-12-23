from resunetmt import ResNetUNetMultiTask
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
from tqdm import tqdm
import gzip
import pickle
import torch


def preprocess_image(image):
    image = cv2.resize(image, (512, 512))
    image = transforms.ToTensor()(image).unsqueeze(0)
    return image.to('cuda')

        
def do_submission(model):
    with gzip.open('task3/test.pkl', 'rb') as f:
        data = pickle.load(f)

    #with gzip.open('task3/train.pkl', 'rb') as f:
    #    pred_data = pickle.load(f)

    model.eval()
    predictions_f = []

    with torch.no_grad():
        for d in tqdm(data):
            prediction_for_video = []
            for frame_idx in range(d['video'].shape[2]):
                frame0 = d['video'][:,:,frame_idx]
                frame0 = cv2.cvtColor(frame0, cv2.COLOR_GRAY2BGR)
                frame = preprocess_image(frame0)
                outputs = model.forward(frame)
                #outputs = torch.sigmoid(outputs)
                prediction = (outputs.cpu().numpy() > 0.05).astype(np.uint8)
                prediction = np.squeeze(prediction, axis=0).transpose(1, 2, 0)
                prediction = cv2.resize(prediction, (frame0.shape[1], frame0.shape[0]))

                prediction_for_video.append(prediction)
                prediction = cv2.cvtColor(prediction, cv2.COLOR_GRAY2BGR)
                stacked_img = np.hstack((frame0, prediction * 255))
                cv2.imwrite(f'inf/frame{frame_idx}.png', stacked_img.astype(np.uint8))

            #print(np.stack(prediction_for_video, axis=-1).shape)
            predictions_f.append({
                'name': d['name'],
                'prediction': np.stack(prediction_for_video, axis=-1).astype(bool)
                }
            )
        
    with gzip.open('task3/submission_pt4.pkl', 'wb') as f:
        pickle.dump(predictions_f, f, 2)
    


model = ResNetUNetMultiTask(n_classes=1).to('cuda')
model.load_state_dict(torch.load('st_output_dir/model_epoch_2.pth'))

do_submission(model)

