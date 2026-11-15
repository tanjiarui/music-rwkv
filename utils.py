import re, subprocess, numpy, torch


def remove_timestamp(lines):
	clean_lines = [re.sub(r'\[\d{2}:\d{2}\.\d{2}]', str(), line) for line in lines]
	return '\n'.join([line.strip() for line in clean_lines if line.strip()])


def load_audio(file, rate=16000, mono=True, max_duration_sec=300):
	cmd = [
		'ffmpeg',
		'-i', file,
		'-f', 'f32le',
		'-acodec', 'pcm_f32le',
		'-ar', str(rate),
		'-ac', '1' if mono else '2',
		'-'
	]
	try:
		out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
		audio = torch.from_numpy(numpy.frombuffer(out, dtype=numpy.float32).copy())
		nonzero = (audio != 0).to(torch.float32)
		if nonzero.sum() > 0:
			start = torch.argmax(nonzero)  # 第一个非零点
			end = len(audio) - torch.argmax(torch.flip(nonzero, dims=[0]))  # 最后一个非零点 + 1
			audio = audio[start:end]
		else:
			print(f'{file} is invalid')
			return None
		# 截断最大时长
		max_samples = rate * max_duration_sec
		if len(audio) > max_samples:
			audio = audio[:max_samples]
		return audio, rate
	except subprocess.CalledProcessError as error:
		print(error)
		return None