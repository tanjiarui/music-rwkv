import os, torch
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from tqdm import tqdm
from transformers.models.deprecated import JukeboxVQVAE
from muq import MuQ
from utils import load_audio, remove_timestamp


# NOTE:
# 这里保留了原始预处理逻辑，但将数据集路径和表格字段做了更清晰的命名，
# 以便在不公开原始数据集的情况下，读者仍能理解整个预处理流程。
def preprocess_dataset(meta, audio_root, output_dir):
	jukebox = JukeboxVQVAE.from_pretrained('openai/jukebox-1b-lyrics')
	muq = MuQ.from_pretrained('OpenMuQ/MuQ-large-msd-iter').model
	os.makedirs(os.path.join(output_dir, 'training'), exist_ok=True)
	os.makedirs(os.path.join(output_dir, 'validation'), exist_ok=True)
	for _, row in tqdm(meta.iterrows(), total=len(meta), desc='tokenization'):
		track_id = str(row['track id'])
		genre = str(row['genre'])
		split = str(row['split'])
		artist = row['artist']
		audio_path = os.path.join(audio_root, genre, f'{track_id}.mp3')
		lyric_path = os.path.join(audio_root, genre, f'{track_id}.lrc')
		target_path = os.path.join(output_dir, split, track_id)

		# 1) 只处理存在的音频文件，并避免重复生成缓存
		if not os.path.exists(audio_path):
			continue
		if os.path.exists(target_path):
			continue

		# 2) 读取歌词并去掉时间戳
		if os.path.exists(lyric_path):
			with open(lyric_path, 'r', encoding='utf-8') as f:
				lyric = remove_timestamp(f.read().strip().split('\n'))
		else:
			lyric = 'none'

		# 3) 加载音频并做采样/裁剪处理
		result = load_audio(audio_path, rate=44100)
		if not result:
			continue
		waveform, _ = result
		waveform = waveform.unsqueeze(0)

		# 4) 对齐到梅尔特征的 hop 大小，便于后续 tokenization
		target_multiple = 4 * 240
		trimmed_length = (waveform.shape[1] // target_multiple) * target_multiple
		tokens, _ = muq.get_targets(waveform[:, :trimmed_length])

		# 5) 编码音频并记录 metadata
		encoded = jukebox.encode(waveform.unsqueeze(0).permute(0, 2, 1).float(), start_level=2, end_level=3)[0].squeeze()
		sample = {
			'track id': track_id,
			'waveform': encoded,
			'spectrogram': tokens['melspec_2048'].squeeze(),
			'meta': {
				'genre': genre.replace('rb', 'rhythm and blues'),
				'artist': artist,
				'lyric': lyric
			}
		}
		torch.save(sample, target_path)


if __name__ == '__main__':
	# 用户可以在这里提供自己的 metadata 表和数据目录。
	# 例如：
	# meta = pd.read_csv('metadata.csv')
	# preprocess_dataset(meta, audio_root='/path/to/audio', output_dir='dataset')
	pass