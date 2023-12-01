# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de


# import joblib
import argparse
from tqdm import tqdm
# import json
# import os.path as osp
import os
import sys
# sys.path.append('.')

import torch
# from human_body_prior.tools.omni_tools import copy2cpu as c2c
# from human_body_prior.body_model.body_model import BodyModel
# from src.datasets import smpl_utils
# from src import config
import numpy as np
# from PIL import Image

import smplx
import pathlib
import clip
import torch.nn.functional as F
from AvatarCLIP.AvatarAnimate.models.render import render_one_batch


comp_device = torch.device("cuda:0")
clip_model, _ = clip.load("ViT-B/32", comp_device)
clip_model = clip_model.eval()

SMPLH_JOINT_NAMES = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_index1',
    'left_index2',
    'left_index3',
    'left_middle1',
    'left_middle2',
    'left_middle3',
    'left_pinky1',
    'left_pinky2',
    'left_pinky3',
    'left_ring1',
    'left_ring2',
    'left_ring3',
    'left_thumb1',
    'left_thumb2',
    'left_thumb3',
    'right_index1',
    'right_index2',
    'right_index3',
    'right_middle1',
    'right_middle2',
    'right_middle3',
    'right_pinky1',
    'right_pinky2',
    'right_pinky3',
    'right_ring1',
    'right_ring2',
    'right_ring3',
    'right_thumb1',
    'right_thumb2',
    'right_thumb3',
    'nose',
    'right_eye',
    'left_eye',
    'right_ear',
    'left_ear',
    'left_big_toe',
    'left_small_toe',
    'left_heel',
    'right_big_toe',
    'right_small_toe',
    'right_heel',
    'left_thumb',
    'left_index',
    'left_middle',
    'left_ring',
    'left_pinky',
    'right_thumb',
    'right_index',
    'right_middle',
    'right_ring',
    'right_pinky',
]



dict_keys = ['betas', 'dmpls', 'gender', 'mocap_framerate', 'poses', 'trans']
action2motion_joints = [8, 1, 2, 3, 4, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 21, 24, 38]  # [18,]


def get_joints_to_use(args):
    joints_to_use = np.array([
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 37
    ])  # 23 joints + global_orient # 21 base joints + left_index1(22) + right_index1 (37)
    return np.arange(0, len(SMPLH_JOINT_NAMES) * 3).reshape((-1, 3))[joints_to_use].reshape(-1)


amass_test_split = ['Transitions_mocap', 'SSM_synced']
amass_vald_split = ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh']
amass_train_split1 = ['CMU', 'MPI_Limits', 'TotalCapture']
amass_train_split2 = ['Eyes_Japan_Dataset', 'KIT']
amass_train_split3 = ['BioMotionLab_NTroje']
amass_train_split4 = ['TCD_handMocap', 'EKUT', 'ACCAD']
amass_train_split5 = ['DFaust_67', 'BMLhandball']
amass_train_split6 = ['BMLmovi']
amass_train_split7 = ['DanceDB']
# Source - https://github.com/nghorbani/amass/blob/08ca36ce9b37969f72d7251eb61564a7fd421e15/src/amass/data/prepare_data.py#L235
amass_splits = {
    'test': amass_test_split,
    'vald': amass_vald_split,
    'train1': amass_train_split1,
    'train2': amass_train_split2,
    'train3': amass_train_split3,
    'train4': amass_train_split4,
    'train5': amass_train_split5,
    'train6': amass_train_split6,
    'train7': amass_train_split7,
}


def pose_padding(pose):
    assert pose.shape[-1] == 69 or pose.shape[-1] == 63
    if pose.shape[-1] == 63:
        padded_zeros = torch.zeros_like(pose)[..., :6]
        pose = torch.cat((pose, padded_zeros), dim=-1)
    return pose


def get_pose_feature(pose, smpl_model):
    # derive from https://github.com/hongfz16/AvatarCLIP/blob/main/AvatarAnimate/models/pose_generation.py
    pose = pose_padding(pose)
    if len(pose.shape) == 1:
        pose = pose.unsqueeze(0)
    bs = pose.shape[0]
    # fix the orientation
    global_orient = torch.zeros(bs, 3).type_as(pose)
    global_orient[:, 0] = np.pi / 2
    output = smpl_model(
        body_pose=pose,
        global_orient=global_orient)
    v = output.vertices
    f = smpl_model.faces
    f = torch.from_numpy(f.astype(np.int32)).unsqueeze(0).repeat(bs, 1, 1).to(self.device)
    angles = (120, 150, 180, 210, 240)
    images = render_one_batch(v, f, angles, comp_device, deterministic=True)   # revise avatarclip code
    images = F.interpolate(images, size=224)
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std = np.array([0.26862954, 0.26130258, 0.27577711])
    images -= torch.from_numpy(mean).reshape(1, 3, 1, 1).to(self.device)
    images /= torch.from_numpy(std).reshape(1, 3, 1, 1).to(self.device)
    num_camera = len(angles)
    image_embed = clip_model.encode_image(images).float().view(num_camera, -1, 512)
    return image_embed.permute(1, 0, 2)   


def process_data(input_folder, output_folder, split_name, joints_to_use, smpl_model):
    sequences = amass_splits[split_name]

    for seq_name in sequences:
        print(f"Reading {seq_name} sequence ...")
        seq_folder = os.path.join(input_folder, seq_name)

        process_single_sequence(seq_folder, output_folder, joints_to_use, smpl_model)


def process_single_sequence(folder, output_folder, joints_to_use, smpl_model):
    subjects = os.listdir(folder)

    for subject in tqdm(subjects):
        if subject.endswith('.txt'):
            continue
        actions = [x for x in os.listdir(osp.join(folder, subject)) if x.endswith('.npz')]

        for action in actions:
            fname = osp.join(folder, subject, action)
            if fname.endswith('shape.npz'):
                continue

            action_dict = dict()
            print('Process '+fname)

            file_name = '/'.join(fanme.split('/')[-3:])

            target_dir = output_folder
            target_file = os.path.join(target_dir, file_name.replace('npz', 'pth'))

            if os.path.exists(target_file):
                print('Exist')
                continue

            pathlib.Path(os.path.split(target_file)[0]).mkdir(parents=True, exist_ok=True)
            data = np.load(fname)

            action_dict['poses'] = data['poses'][:, joints_to_use]
            action_dict['mocap_framerate'] = data['mocap_framerate']

            clip_features = []
            with torch.no_grad():
                for ind in tqdm(range(0, len(action_dict['poses']), 50)):
                    clip_features.append(get_pose_feature(torch.FloatTensor(action_dict['poses'][ind: ind + 50]).to(comp_device), smpl_model).detach())
            action_dict['clip_features'] = torch.cat(clip_features, 0)

            torch.save(action_dict, target_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help='dataset directory', default='./data/amass')
    parser.add_argument('--output_dir', type=str, help='target directory', default='./data/amass_clip_features')
    parser.add_argument('--smpl_path', type=str, help='dataset directory', default='./AvatarCLIP/smpl_models')
    parser.add_argument('--split_name', type=str, default='test')
    
    args = parser.parse_args()

    smpl_model = smplx.create(args.smpl_path, 'smpl').to(comp_device).eval()

    joints_to_use = get_joints_to_use(args)
    
    process_data(args.input_dir, args.output_dir, split_name=args.split_name, joints_to_use=joints_to_use, smpl_model=smpl_model)
