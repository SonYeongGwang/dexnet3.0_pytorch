import numpy as np
import glob
import os
import tqdm
import copy
import torch
from torch.utils.data import Dataset

'''
note
- DexNet3.0 datasets consist of 2,760 objs and each has 1,000 suction samples
- Input depth image in 32x32x1
- DexNet3.0 dataset download path: https://berkeley.app.box.com/s/6mnb2bzi5zfa7qpwyn7uq5atb7vbztng/folder/38455887072
- To fetch full dexnet3.0 dataset onto your RAM, there should be at least 20GB free space.
'''

home = os.path.expanduser('~')
def getCandidateInfoList(number_of_files = 1000,use_sub_data = True):
    
    data_path = home+"your/path/to/tensors"
    ## for exmple) "/Documents/dexnet_3/dexnet_09_13_17/tensors"

    depth_img_list = glob.glob(data_path+'/depth*')
    hand_poses_list = glob.glob(data_path+'/hand*')
    grasp_metric_list = glob.glob(data_path+'/robust*')

    ## sort array to match orders among the lists
    depth_img_list.sort()
    hand_poses_list.sort()
    grasp_metric_list.sort()
    
    file_list = [i for i in range(0, number_of_files)]

    if use_sub_data:
        depth_img_list = [depth_img_list[i] for i in file_list]
        hand_poses_list = [hand_poses_list[i] for i in file_list]
        grasp_metric_list = [grasp_metric_list[i] for i in file_list]
    
    return depth_img_list, hand_poses_list, grasp_metric_list

class SuctionDataset:
    def __init__(self, val_stride=0, isValSet_bool=None, number_of_files=1000):
        print("INFO: Initiating SuctionDataset...")
        self.depth_img_list, self.hand_poses_list, self.grasp_metric_list = getCandidateInfoList(use_sub_data=True, number_of_files=number_of_files)
        self.loadRawData()
        self.total_data_index = [i for i in range(1000*number_of_files)]

    def loadRawData(self):
        ## load dexnet3.0 dataset onto RAM
        ## the number of dataset can be selected with 'number_of_files'
        print("INFO: loading dataset...")
        self.depth_imgs = []
        self.hand_poses = []
        self.grasp_metrics = []

        for key in tqdm.tqdm(zip(self.depth_img_list, self.hand_poses_list, self.grasp_metric_list), total=len(self.depth_img_list)):
            self.depth_imgs.append(np.load(key[0])['arr_0'])
            self.hand_poses.append(np.load(key[1])['arr_0'])
            self.grasp_metrics.append(np.load(key[2])['arr_0'])

class TrainingSuctionDataset(Dataset):
    def __init__(self, dataset, val_stride):
        print("INFO: Initiating TrainingSuctionDataset...")
        self.total_data_index = copy.copy(dataset.total_data_index)
        del (self.total_data_index[::val_stride])

        self.depth_imgs = dataset.depth_imgs
        self.hand_poses = dataset.hand_poses
        self.grasp_metrics = dataset.grasp_metrics
    def __len__(self):
        return len(self.total_data_index)

    def __getitem__(self, index: int):
        ## To select between seperate files
        i = self.total_data_index[index]
        file_index = i//1000
        item_index = i-1000*(i//1000)
        self.depth_img = self.depth_imgs[file_index][item_index]
        self.hand_pose = self.hand_poses[file_index][item_index]
        self.grasp_metric = self.grasp_metrics[file_index][item_index]
        ## torch expect image to be C × H × W
        ## reshape depth_image from (32, 32, 1) to (1, 32, 32)
        self.depth_img = np.reshape(self.depth_img, (1, 32, 32))

        ## grasp is considered as success if the grasp_metric >= 0.2
        self.grasp_metric = 1 if self.grasp_metric >= 0.2 else 0

        ## convert numpy array to torch.tensor
        self.depth_img_t = torch.from_numpy(self.depth_img)
        self.hand_pose_t = torch.from_numpy(self.hand_pose[[2, 3]])
        self.grasp_metric_t = torch.tensor(self.grasp_metric, dtype=torch.int64)

        return (self.depth_img_t, self.hand_pose_t, self.grasp_metric_t)

class ValidationSuctionDataset(Dataset):
    def __init__(self, dataset, val_stride):
        print("INFO: Initiating ValidationSuctionDataset...")
        self.total_data_index = dataset.total_data_index[::val_stride]

        self.depth_imgs = dataset.depth_imgs
        self.hand_poses = dataset.hand_poses
        self.grasp_metrics = dataset.grasp_metrics
    def __len__(self):
        return len(self.total_data_index)

    def __getitem__(self, index: int):
        ## To select between seperate files
        i = self.total_data_index[index]
        file_index = i//1000
        item_index = i-1000*(i//1000)
        self.depth_img = self.depth_imgs[file_index][item_index]
        self.hand_pose = self.hand_poses[file_index][item_index]
        self.grasp_metric = self.grasp_metrics[file_index][item_index]
        ## torch expect image to be C × H × W
        ## reshape depth_image from (32, 32, 1) to (1, 32, 32)
        self.depth_img = np.reshape(self.depth_img, (1, 32, 32))

        ## grasp is considered as success if the grasp_metric >= 0.2
        self.grasp_metric = 1 if self.grasp_metric >= 0.2 else 0

        ## convert numpy array to torch.tensor
        self.depth_img_t = torch.from_numpy(self.depth_img)
        self.hand_pose_t = torch.from_numpy(self.hand_pose[[2, 3]])
        self.grasp_metric_t = torch.tensor(self.grasp_metric, dtype=torch.int64)

        return (self.depth_img_t, self.hand_pose_t, self.grasp_metric_t)

if __name__ == "__main__":
    SD = SuctionDataset(val_stride=10, isValSet_bool=True, number_of_files=100)
    TSD = TrainingSuctionDataset(SD, val_stride=3)
    VSD = ValidationSuctionDataset(SD, val_stride=3)
    print(TSD[0][0].size())
