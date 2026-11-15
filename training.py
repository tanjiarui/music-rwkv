import json, pandas as pd, torch.nn.functional as F
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler
from module import *
from model import MODEL
torch.backends.cudnn.benchmark = True
max_length = {'waveform': 20000, 'spectrogram': 10000, 'meta': 500}
token_size = json.load(open('token size', 'r'))
scaler = GradScaler()


class MusicDataset(Dataset):
	def __init__(self, folder, mode):
		assert mode in ['train', 'validation'], 'invalid mode'
		self.mode = mode
		self.files = [os.path.join(folder, file) for file in os.listdir(folder)]

	def __len__(self):
		return len(self.files)

	def __getitem__(self, index) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
		x = torch.load(str(self.files[index]), map_location='cpu')
		if 'track id' in x:
			x.pop('track id')
		y = x.copy()
		# tokenize meta
		masked_fields = dict()
		for field, text in x['meta'].items():
			masked_fields[field] = tokenizer.encode(str(text))
		meta_input_ids = list()
		for field in x['meta']:
			prefix = tokenizer.encode(f'{field}: ')
			meta_input_ids.extend(prefix + masked_fields[field] + tokenizer.encode('\n'))
		x['meta'] = torch.tensor(meta_input_ids, dtype=torch.int64, device='cpu')
		masked_fields = dict()
		for field, text in y['meta'].items():
			masked_fields[field] = tokenizer.encode(str(text))
		meta_input_ids = list()
		for field in y['meta']:
			prefix = tokenizer.encode(f'{field}: ')
			meta_input_ids.extend(prefix + masked_fields[field] + tokenizer.encode('\n'))
		y['meta'] = torch.tensor(meta_input_ids, dtype=torch.int64, device='cpu')
		# 对每个键进行截断
		for data in [x, y]:
			for modality in data:
				if modality in max_length:
					pad_len = max_length[modality] - data[modality].shape[0]
					if pad_len <= 0:
						data[modality] = data[modality][:max_length[modality]]
		return x, y


# 按照当前批次中每个模态的最大长度进行填充和堆叠
def collate_data(batch):
	xs = [s[0] for s in batch]
	ys = [s[1] for s in batch]
	max_len = dict()
	for modality in token_size.keys():
		length = list()
		for t in xs + ys:
			tt = t[modality]
			length.append(tt.size(-1) if tt.dim() >= 2 else tt.size(0))
		max_len[modality] = max(length) if length else 0

	# ensure max lengths are multiples of both strides (lcm)
	# compute lcm safely
	l = math.lcm(max(1, waveform_stride), max(1, spectrogram_stride))
	if l > 1:
		for modality in list(max_len.keys()):
			max_len[modality] = ((max_len[modality] + l - 1) // l) * l

	def pad_and_stack(dicts):
		out = dict()
		for m in token_size.keys():
			fill_val = token_size[m] - 1
			padded = list()
			lengths = list()
			for d in dicts:
				if d[m].dim() == 1:
					length_m = d[m].size(0)
					pad_len = max_len[m] - length_m
					if pad_len > 0:
						pad = torch.full((pad_len,), fill_val, dtype=torch.int64, device='cpu')
						d[m] = torch.cat([d[m], pad], dim=0)
					lengths.append(length_m)
				elif d[m].dim() == 2:
					channel, sequence = d[m].size(0), d[m].size(1)
					pad_len = max_len[m] - sequence
					if pad_len > 0:
						pad = torch.full((channel, pad_len), fill_val, dtype=torch.int64, device='cpu')
						d[m] = torch.cat([d[m], pad], dim=1)
					lengths.append(sequence)
				else:
					raise ValueError(f'unsupported dim {d[m].dim()} for modality {m}')
				padded.append(d[m])
			out[m] = torch.stack(padded, dim=0)  # [B, ...]
			out[f'{m} length'] = torch.tensor(lengths, dtype=torch.int64)
			out[f'{m} mask'] = (out[f'{m} length'].unsqueeze(1) > torch.arange(max_len[m]).unsqueeze(0)).to(torch.bool)
		return out

	batched_x = pad_and_stack(xs)
	batched_y = pad_and_stack(ys)
	return batched_x, batched_y


C = 1024
batch_size = 4
epochs = 100
waveform_stride = 10
spectrogram_stride = 10
lambda_factor = 1e-3
accum_steps = 4
train_path = 'dataset/training'
validation_path = 'dataset/validation'

train_set = MusicDataset(train_path, 'train')
validation_set = MusicDataset(validation_path, 'validation')
if is_distributed:
	train_sampler = DistributedSampler(train_set, shuffle=True)
	val_sampler = DistributedSampler(validation_set, shuffle=False)
else:
	train_sampler = None
	val_sampler = None
train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, collate_fn=collate_data, shuffle=(not is_distributed), drop_last=True)
validation_loader = DataLoader(validation_set, batch_size=batch_size, sampler=val_sampler, collate_fn=collate_data, shuffle=False, drop_last=True)
# checkpoint = torch.load('best full', map_location=device)
# 初始化模型
model = MODEL(C, token_size['waveform'], token_size['spectrogram'], token_size['meta'], waveform_stride, spectrogram_stride).to(device)
# model.load_state_dict(checkpoint['model state'])
if is_distributed:
	model_engine = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else list(), find_unused_parameters=True)
else:
	model_engine = model
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# optimizer.load_state_dict(checkpoint['optimizer state'])
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=.5, patience=3, min_lr=1e-6)


def loss_and_metric(prediction: torch.Tensor, target: torch.Tensor, vocab_size: int):
	target_flat = target.view(-1)
	ignore_index = vocab_size - 1
	mask = target_flat != ignore_index
	loss_all = l2wrap_cross_entropy(prediction, target, ignore_index)
	loss = loss_all[mask].mean()
	pred_token = prediction.argmax(dim=-1)  # [batch, seq]
	correct = (pred_token == target) & (target != ignore_index)
	return loss, correct.sum(), (target != ignore_index).sum()


try:
	log = pd.read_excel('log.xlsx', engine='openpyxl')
	if 'training loss' in log.columns and len(log) > 0:
		best_train_loss = float(log['training loss'].min())
		best_validation_loss = float(log['validation loss'].min())
	else:
		best_train_loss = float('inf')
		best_validation_loss = float('inf')
except Exception as e:
	print(e)
	log = pd.DataFrame(columns=['epoch', 'training loss', 'training accuracy', 'validation loss', 'validation accuracy'])
	best_train_loss = float('inf')
	best_validation_loss = float('inf')
for epoch in range(epochs):
	epoch_loss_train = 0
	epoch_correct_train = 0
	epoch_total_train = 0
	epoch_loss_validation = 0
	epoch_correct_validation = 0
	epoch_total_validation = 0
	model_engine.train()
	if is_distributed and isinstance(train_sampler, DistributedSampler):
		train_sampler.set_epoch(epoch)
	for step, (x_train, y_train) in enumerate(tqdm(train_loader, desc=f'[rank {rank}] training', unit='batch')):
		x_train: dict[str, torch.Tensor]
		for key in x_train:
			x_train[key] = x_train[key].to(device)
		for key in y_train:
			y_train[key] = y_train[key].to(device)
		with autocast(device_type=device.type, dtype=torch.bfloat16):
			waveform_pred, spectrogram_pred, meta_pred, waveform_recon, spectrogram_recon = model_engine(x_train['waveform'], x_train['spectrogram'], x_train['meta'])
			waveform_original = model.embedding1(x_train['waveform'])  # [B, T1, C]
			spectrogram_original = model.embedding2(x_train['spectrogram'])  # [B, T2, C]
			waveform_loss, waveform_correct, waveform_total = loss_and_metric(waveform_pred, y_train['waveform'], token_size['waveform'])
			spectrogram_loss, spectrogram_correct, spectrogram_total = loss_and_metric(spectrogram_pred, y_train['spectrogram'], token_size['spectrogram'])
			meta_loss, meta_correct, meta_total = loss_and_metric(meta_pred, y_train['meta'], token_size['meta'])
			loss_reconstruct = F.mse_loss(F.layer_norm(waveform_recon.detach(), (C,)), F.layer_norm(waveform_original, (C,))) + F.mse_loss(F.layer_norm(spectrogram_recon.detach(), (C,)), F.layer_norm(spectrogram_original, (C,)))
			final_loss = waveform_loss + spectrogram_loss + meta_loss + lambda_factor * loss_reconstruct
			if not torch.isfinite(final_loss):
				print(f'waveform loss {waveform_loss.item()}\t spectrogram loss {spectrogram_loss.item()}\t meta loss {meta_loss.item()}\t reconstruct loss {loss_reconstruct.item()}')
			scaler.scale(final_loss / accum_steps).backward()
			if (step + 1) % accum_steps == 0:
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
				scaler.step(optimizer)
				scaler.update()
				optimizer.zero_grad()

		epoch_correct_train += waveform_correct.item() + spectrogram_correct.item() + meta_correct.item()
		epoch_total_train += waveform_total.item() + spectrogram_total.item() + meta_total.item()
		epoch_loss_train += final_loss.item()
		if step % 100 == 0 and step != 0 and rank == 0:
			print(f'checkpoint at step {step} (skipped saving — saving only when both train+validation losses improve)')
	if rank == 0:
		print(f'epoch {epoch}\ntraining loss = {epoch_loss_train / len(train_loader):.4f}, training accuracy = {epoch_correct_train / epoch_total_train * 100:.2f}%')
	model_engine.eval()
	with torch.no_grad():
		for x_validation, y_validation in tqdm(validation_loader, desc=f'[rank {rank}] validation', unit='batch'):
			for key in x_validation:
				x_validation[key] = x_validation[key].to(device)
			for key in y_validation:
				y_validation[key] = y_validation[key].to(device)
			with autocast(device_type=device.type, dtype=torch.bfloat16):
				waveform_pred, spectrogram_pred, meta_pred, waveform_recon, spectrogram_recon = model_engine(x_validation['waveform'], x_validation['spectrogram'], x_validation['meta'])
				waveform_original = model.embedding1(x_validation['waveform'])  # [B, T1, C]
				spectrogram_original = model.embedding2(x_validation['spectrogram'])  # [B, T2, C]
				waveform_loss, waveform_correct, waveform_total = loss_and_metric(waveform_pred, y_validation['waveform'], token_size['waveform'])
				spectrogram_loss, spectrogram_correct, spectrogram_total = loss_and_metric(spectrogram_pred, y_validation['spectrogram'], token_size['spectrogram'])
				meta_loss, meta_correct, meta_total = loss_and_metric(meta_pred, y_validation['meta'], token_size['meta'])
				loss_reconstruct = F.mse_loss(F.layer_norm(waveform_recon.detach(), (C,)), F.layer_norm(waveform_original, (C,))) + F.mse_loss(F.layer_norm(spectrogram_recon.detach(), (C,)), F.layer_norm(spectrogram_original, (C,)))
				final_loss = waveform_loss + spectrogram_loss + meta_loss + lambda_factor * loss_reconstruct

			epoch_correct_validation += waveform_correct.item() + spectrogram_correct.item() + meta_correct.item()
			epoch_total_validation += waveform_total.item() + spectrogram_total.item() + meta_total.item()
			epoch_loss_validation += final_loss.item()
	if rank == 0:
		print(f'validation loss = {epoch_loss_validation / len(validation_loader):.4f}, validation accuracy = {epoch_correct_validation / epoch_total_validation * 100:.2f}%')
		# save only if both training and validation loss improved
		curr_train = epoch_loss_train / len(train_loader)
		curr_val = epoch_loss_validation / len(validation_loader)
		scheduler.step(curr_val)
		if curr_train < best_train_loss and curr_val < best_validation_loss:
			torch.save({'epoch': epoch, 'model state': model.state_dict(), 'optimizer state': optimizer.state_dict(), 'training': {'loss': curr_train, 'accuracy': epoch_correct_train / epoch_total_train},
			            'validation': {'loss': curr_val, 'accuracy': epoch_correct_validation / epoch_total_validation}}, os.path.join('best full'))
			best_train_loss = curr_train
			best_validation_loss = curr_val
		else:
			print(f'not saving model. curr_train={curr_train:.6f} (best {best_train_loss:.6f}), curr_val={curr_val:.6f} (best {best_validation_loss:.6f})')
		current = pd.DataFrame({'epoch': [epoch], 'training loss': [curr_train], 'training accuracy': [epoch_correct_train / epoch_total_train], 'validation loss': [curr_val], 'validation accuracy': [epoch_correct_validation / epoch_total_validation]})
		log = pd.concat([log, current], ignore_index=True)
		log.to_excel('log.xlsx', index=False, engine='openpyxl')