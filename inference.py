import os, json, torch, torchaudio
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers.models.deprecated import JukeboxVQVAE
from muq import MuQ
from torch.amp import autocast, GradScaler
from utils import remove_timestamp, load_audio
from module import device, tokenizer
from model import MODEL
torch.backends.cudnn.benchmark = True
max_length = {'waveform': 20000, 'spectrogram': 10000, 'meta': 500}
token_size = json.load(open('token size', 'r'))
scaler = GradScaler()


def pad_to_multiple(x, multiple=10, pad_value=0):
	if x.dim() == 0:
		return x
	length = x.shape[0]
	remainder = length % multiple
	if remainder == 0:
		return x
	return torch.nn.functional.pad(x, (0, multiple - remainder), mode='constant', value=pad_value)


audio_path = 'red'
lyric_path = 'lyric'
genre = 'country'
artist = 'Taylor Swift'
checkpoint = torch.load('autodl-fs/best full', map_location=device)
jukebox = JukeboxVQVAE.from_pretrained('openai/jukebox-1b-lyrics')
muq = MuQ.from_pretrained('OpenMuQ/MuQ-large-msd-iter').model
lyric = remove_timestamp(open(lyric_path, 'r').read().strip().split('\n'))

result = load_audio(audio_path, rate=44100)
if not result:
	exit(0)
waveform, sr = result
waveform = waveform.unsqueeze(0)
target_multiple = 4 * 240  # 240 是梅尔帧 hop size（通常是 10ms 对应的采样点数）
trimmed_length = (waveform.shape[1] // target_multiple) * target_multiple
tokens, _ = muq.get_targets(waveform[:, :trimmed_length])
data = {'waveform': jukebox.encode(waveform.unsqueeze(0).permute(0, 2, 1).float(), start_level=2, end_level=3)[0].squeeze(), 'spectrogram': tokens['melspec_2048'].squeeze(), 'meta': {'genre': genre, 'artist': artist, 'lyric': lyric}}

# tokenize meta
masked_fields = dict()
for field, text in data['meta'].items():
	masked_fields[field] = tokenizer.encode(str(text))
meta_input_ids = list()
for field in data['meta']:
	prefix = tokenizer.encode(f'{field}: ')
	meta_input_ids.extend(prefix + masked_fields[field] + tokenizer.encode('\n'))
data['meta'] = torch.tensor(meta_input_ids, dtype=torch.int64, device='cpu')
# 对每个键进行截断
for modality in data:
	if modality in max_length:
		pad_len = max_length[modality] - data[modality].shape[0]
		if pad_len <= 0:
			data[modality] = data[modality][:max_length[modality]]
for modality in ('waveform', 'spectrogram'):
	if modality in data:
		data[modality] = pad_to_multiple(data[modality], multiple=10, pad_value=token_size[modality] - 1)

# 初始化模型
model = MODEL(1024, token_size['waveform'], token_size['spectrogram'], token_size['meta'], 10, 10).to(device)
model.load_state_dict(checkpoint['model state'])
model.eval()
with torch.no_grad():
	for key in data:
		data[key] = data[key].to(device)
	with autocast(device_type=device.type, dtype=torch.bfloat16):
		waveform_pred, spectrogram_pred, meta_pred, waveform_recon, spectrogram_recon = model(data['waveform'].unsqueeze(dim=0), data['spectrogram'].unsqueeze(dim=0), data['meta'].unsqueeze(dim=0))
waveform_tokens = waveform_pred.argmax(dim=-1)

recovery = jukebox.decode(waveform_tokens.to('cpu').unsqueeze(-1), start_level=2, end_level=3)
torchaudio.save('recovery.wav', recovery.squeeze(-1).reshape(-1).unsqueeze(0), 44100)
original = jukebox.decode(data['waveform'][data['waveform'] != 2048].to('cpu').unsqueeze(dim=0).unsqueeze(dim=-1), start_level=2, end_level=3)
torchaudio.save('original.wav', original.squeeze(-1).reshape(-1).unsqueeze(0), 44100)