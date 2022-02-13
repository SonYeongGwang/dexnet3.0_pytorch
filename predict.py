import os
import numpy as np
import torch
import torch.nn as nn
import time
import cv2

home = os.path.expanduser('~')

from model import SuctionModel

from collections import OrderedDict


trained_parameters_path = "./trained_model"

device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
print("INFO: Predicting with {}".format(device))

t = time.time()
sm = SuctionModel()
print("INFO: Loading the model to the {}".format(device))
sm = sm.to(device=device)
print("INFO: Loading the model to the {} took {:.3f}s".format(device, time.time()-t))

t = time.time()
print("INFO: Loading the pretrained prameters")
state_dict = torch.load(trained_parameters_path+"/DexNet_25epochs_full_dataset_75-25_train_val_ratio.pt")
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
## load params
sm.load_state_dict(new_state_dict)
print("INFO: Loading the pretrained prameters took {:.3f}s".format(time.time()-t))

sm.eval()

## load test image and grasp information
data_path = home+"your/path/to/tensors"
data_path = home+"/Documents/dexnet_3/dexnet_09_13_17/tensors"

file_index = '00018'
interest = 1

depth_image = np.load(data_path+'/depth_ims_tf_table_'+file_index+'.npz')['arr_0'][interest,...]
hand_pose = np.load(data_path+'/hand_poses_'+file_index+'.npz')['arr_0'][interest,...]
grasp_metric = np.load(data_path+'/robust_suction_wrench_resistance_'+file_index+'.npz')['arr_0'][interest,...]

width = np.shape(depth_image)[0]
height = np.shape(depth_image)[1]
depth_image = np.reshape(depth_image, (width, height))
depth_resized = cv2.resize(depth_image, dsize=(width*5, height*5), interpolation=cv2.INTER_AREA)

## predict with the trained GQ-CNN
depth_img_t = torch.from_numpy(depth_image)
hand_pose_t = torch.from_numpy(hand_pose[[2, 3]])

depth_img_t = depth_img_t.to(device).float()
hand_pose_t = hand_pose_t.to(device).float()

out = sm(depth_img_t, hand_pose_t)
soft_max = nn.Softmax(-1)
result = soft_max(out).argmax().item()

prediction = "success" if result==1 else "fail"
ground_truth = "success" if grasp_metric>=0.2 else "fail"

print("The model says:", prediction)
print("The ground truth:", ground_truth)

cv2.circle(depth_resized, (np.shape(depth_resized)[0]//2, np.shape(depth_resized)[1]//2), 1, (0, 0, 0), 2)
cv2.imshow("depth", depth_resized)
cv2.waitKey(0)