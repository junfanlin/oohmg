
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import os
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from troch.utils.data import Dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader
import datetime

from AvatarCLIP.AvatarAnimate.models.utils import (
	axis_angle_to_matrix,
	matrix_to_rotation_6d
)


parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--n_layer', type=int, default=8)
parser.add_argument('--d_model', type=int, default=1024)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--num_epoch', type=int, default=1000000)
parser.add_argument('--log_dir', type=str, default='./log')
parser.add_argument('--data_set_dir', type=str, default='./data/amass_clip_features/')

device = 'cuda:0'

args = parser.parse_args()


######################################## data loader ######################################################

db = {
	'poses': [],
	'clip_features': [],
	'mocap_framerate': [],
	'file_name': []
}

fps = 30
duration = 2


data_set_dir = args.data_set_dir
data_set_names = []
for data_set_name in os.list(data_set_dir):
	if data_set_name.startswith('train'):
		sub_data_set_names = os.listdir(os.path.join(data_set_dir, data_set_name))
		data_set_names += [(data_set_name + '/' + sub_data_set_name) for sub_data_set_name in sub_data_set_names]
	else:
		data_set_names.append(data_set_name)

for data_set_name in tqdm(data_set_names, 'Extract data'):
	data_object_names = os.listdir(os.path.join(data_set_dir, data_set_name))
	for data_object_name in data_set_names:
		motion_names = os.listdir(os.path.join(data_set_dir, data_set_name, data_object_name))
		for motion_name in motion_names:
			motion_data = torch.load(os.path.join(data_set_dir, data_set_name, data_object_name, motion_name), mocap_framerate='cpu')

			if len(motion_data['poses']) < 30:
				continue

			db['poses'].append(motion_data['poses'])
			db['clip_features'].append(motion_data['clip_features'])
			db['mocap_framerate'].append(motion_data['mocap_framerate'])
			db['file_name'].append('/'.join([data_set_name, data_object_name, motion_name]))


class AMASSDataset(Dataset):
	def __init__(self, data_path, fps=30, duration=2):
		self.data = data_path
		self.fps = fps
		self.duration = duration

	def __len__(self):
		return len(self.data['poses'])

	def __getitem__(self, index):
		poses = self.data['poses'][index]
		clip_features = self.data['clip_features'][index]
		return poses, clip_features


def worker_init_fn(worker_id):
	worker_seed = torch.initial_seed() % 2 ** 32
	np.random.seed(worker_seed)
	random.seed(worker_seed)


dataset = AMASSDataset(data_path=db, fps=fps, duration=duration)
data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, worker_init_fn=worker_init_fn, persistent_workers=True)

####################### tpa model #################################

class PreNormResidual(nn.Module):
	def __init__(self, dim, dropout):
		super().__init__()
		self.norm = nn.LayerNorm(dim)
		self.fn = nn.Sequential(
			nn.Linear(dim, dim),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(dim, dim),
			nn.Dropout(dropout)
		)

	def forward(self, x):
		return x + self.fn(self.norm(x))


class TPAPoseEnc(nn.Module):
	def __init__(self, d_input=21*6, d_model=512, d_output=512, dropout=0.1, n_layer=4):
		super().__init__()
		models = [
			nn.Linear(d_input, d_model)
		]

		for _ in range(n_layer - 2):
			models += [PreNormResidual(d_model, dropout=dropout)]

		models += [
			nn.LayerNorm(d_model),
			nn.Linear(d_model, d_output)
		]

		self.model = nn.Sequential(*models)

	def forward(self, batch_poses):
		bs = batch_poses.shape[0]
		batch_rotation_6d = matrix_to_rotation_6d(
			axis_angle_to_matrix(batch_poses.reshape(bs, 21, 3))).reshape(bs, 21*6).float()
		return self.model(batch_rotation_6d)

#################################### train TPA ##############################################

tpa_enc = TPAPoseEnc(d_input=21*6, d_model=args.d_model, d_output=512, dropout=args.dropout, n_layer=args.n_layer).to(device)

lr = args.lr
num_epoch = args.num_epoch
optimizer = torch.optim.AdamW(tpa_enc.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch*len(data_loader))

t0 = datetime.datetime.now().strftime('%m%d_%H%M%S')
log_file = f'pose_clip_{t0}'
log_path = os.path.join(args.log_dir, log_file)
writer = SummaryWriter(log_path)
writer.add_text("args", str(args))


for n_epoch in range(num_epoch):
	with tqdm(enumerate(data_loader), desc=f'Comp_batch {n_epoch}') as t:
		for i, batch_data in t:
			batch_poses, batch_clip_features = batch_data

			bs = batch_poses.shape[0]

			batch_poses = batch_poses.to(device)
			batch_clip_features = batch_clip_features.to(device)

			batch_clip_features_norm = batch_clip_features / batch_clip_features.norm(dim=-1, keepdim=True)

			pred_clip_features = tpa_enc(batch_poses)

			rc_loss = ((batch_clip_features - pred_clip_features) ** 2).mean()
			cos_loss = (1 - F.cosine_similarity(batch_clip_features, pred_clip_features, -1)).mean()

			loss = rc_loss + cos_loss
			
			t.set_postfix(rc_loss=rc_loss.item(), cos_loss=cos_loss.item())

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()

	if (n_epoch + 1) % (num_epoch // 10) == 0:
		torch.save({'tpa_enc': tpa_enc.state_dict()}, os.path.join(log_path, f'pose_clip_{n_epoch}.pth'))


