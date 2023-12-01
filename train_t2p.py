import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch import nn

from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser
from AvatarCLIP.AvatarAnimate.models.utils import (
	axis_angle_to_matrix,
	matrix_to_rotation_6d
)

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import datetime
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--n_layer', type=int, default=8)
parser.add_argument('--d_model', type=int, default=1024)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--num_epoch', type=int, default=1000)
parser.add_argument('--num_iterations', type=int, default=1000)
parser.add_argument('--use_latent_reg', type=int, default=1)
parser.add_argument('--latent_reg_weight', type=float, default=0.15)
parser.add_argument('--temperature', type=float, default=0.05)
parser.add_argument('--log_dir', type=str, default='./log')
parser.add_argument('--tpa_path', type=str, default='./log/pose_clip_999999.pth')
parser.add_argument('--vposer_path', type=str, default='./data/vposer')

device = 'cuda:0'
args = parser.parse_args()

############### load VPoser ############
vposer_path = args.vposer_path
vp, _ = load_model(
	vposer_path,
	model_code=VPoser,
	remove_words_in_model_weights='vp_model.',
	disable_grad=True
)
vp = vp.to(device).eval()
#######################################

############### load TPA ####################

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


tpa_enc = TPAPoseEnc(d_input=21*6, d_model=1024, d_output=512, dropout=0.1, n_layer=8).to(device).eval()
tpa_enc.load_state_dict(torch.load(args.tpa_path)['tpa_enc'])

####################################  train T2P #################################

class TextPoseGen(nn.Module):
	def __init__(self, d_input=32+512, d_model=512, d_output=32, dropout=0.1, n_layer=4):
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

	def forward(self, batch_text_features):
		batch_text_features = batch_text_features / batch_text_features.norm(dim=-1, keepdim=True)
		return self.model(batch_text_features)


t2p_gen = TextPoseGen(d_input=512, d_model=args.d_model, d_output=32, dropout=args.dropout, n_layer=args.n_layer).to(device)
optimizer = torch.optim.AdamW(t2p_gen.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoch*args.num_iterations)

one_hot_emb = torch.eye(512).float().to(device)
temperature = args.temperature

######### log #########
t0 = datetime.datetime.now().strftime('%m%d_%H%M%S')
log_file = f'train_pose_vprior_random_latreg{arg.use_latent_reg*args.latent_reg_weight}_{t0}'
log_path = os.path.join(args.log_dir, log_file)
writer = SummaryWriter(log_path)
writer.add_text("args", str(args))

Path(log_path).mkdir(exist_ok=True, parent=True)
#######################

for epoch in range(args.num_epoch):
	with tqdm(range(args.num_iterations), desc=f'Epoch {epoch} ') as t:
		while t.n < t.total:
			bs = args.batch_size

			text_features = torch.cat(
				[
					torch.randn(bs // 2, 512).to(device),
					(torch.rand(bs // 2, 512) * 2 - 1).to(device)
				], 0)

			text_features_bias = (torch.rand(bs // 2 * 2, 1) * 2 - 1).to(device)
			text_features = text_features - text_features_bias

			pose_latent_enhance = t2p_gen(text_features)

			loss = 0
			loss_to_log = {}

			clip_feature_enhance = tpa_enc(vp.decode(pose_latent_enhance)['pose_body'].reshape(-1, 21*3))

			score = F.cosine_similarity(clip_feature_enhance[:, None], text_features[None], -1) / temperature

			loss_i = -torch.diag(score.log_softmax(dim=0)).mean()
			loss_t = -torch.diag(score.log_softmax(dim=1)).mean()

			loss = (loss_i + loss_t) / 2

			loss_to_log['loss_i'] = loss_i.item()
			loss_to_log['loss_t'] = loss_t.itme()

			if args.use_latent_reg:
				loss_0 = (pose_latent_enhance ** 2).mean()
				loss = loss + loss_0 * args.latent_reg_weight
				loss_to_log['loss_0'] = loss_0.item()

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()

	writer.add_scalar('loss/image', loss_i.item(), epoch)
	writer.add_scalar('loss/text', loss_t.item(), epoch)

	if (epoch + 1) % (args.num_epoch // 10) == 0:
		torch.save(t2p_gen.state_dict(), os.path.join(log_path, f'epoch_{epoch}.pth'))
