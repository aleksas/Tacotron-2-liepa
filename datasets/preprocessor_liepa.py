from os.path import join
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from datasets import audio
import os
from os import listdir
import numpy as np
from wavenet_vocoder.util import mulaw_quantize, mulaw, is_mulaw, is_mulaw_quantize
from os.path import join, splitext, isdir

import codecs
import chardet

def build_from_path(hparams, input_dirs, max_files_per_speaker, mel_dir, linear_dir, wav_dir, n_jobs=12, tqdm=lambda x: x):
	"""
	Preprocesses the speech dataset from a gven input path to given output directories

	Args:
		- hparams: hyper parameters
		- input_dirs: input directories that contain the files to prerocess
		- max_files_per_speaker: max files per speaker
		- mel_dir: output directory of the preprocessed speech mel-spectrogram dataset
		- linear_dir: output directory of the preprocessed speech linear-spectrogram dataset
		- wav_dir: output directory of the preprocessed speech audio dataset
		- n_jobs: Optional, number of worker process to parallelize across
		- tqdm: Optional, provides a nice progress bar

	Returns:
		- A list of tuple describing the train examples. this should be written to train.txt
	"""

	# We use ProcessPoolExecutor to parallelize across processes, this is just for
	# optimization purposes and it can be omited
	executor = ProcessPoolExecutor(max_workers=n_jobs)
	futures = []
	index = 1
	for input_dir in input_dirs:
		for wav_path in collect_files(input_dir, max_files_per_speaker):
			txt_path = wav_path.replace('.wav','.txt')
			text = load_text(txt_path)
			futures.append(executor.submit(partial(_process_utterance, mel_dir, linear_dir, wav_dir, index, wav_path, text, hparams)))
			index += 1

	return [future.result() for future in tqdm(futures) if future.result() is not None]

def load_text(txt_path):
	raw_text = None
	with open(txt_path, 'rb') as f:
		raw_text = f.read()

	result = chardet.detect(raw_text)
	charenc = result['encoding']

	if charenc == 'UTF-16':
	    charenc = "utf-16le"
	elif charenc == 'ISO-8859-1':
	    charenc = 'windows-1257'

	text = None
	with codecs.open(txt_path, encoding=charenc) as fin:
	    text = fin.read()
	    for m in meta:
	        text = text.replace(m, '')
	    for mm_p, mm_r in meta_m:
	        text = text.replace(mm_p, mm_r)
	    text = text.replace('\n', ' ').replace('\r', '').strip().lower()
	return ' '.join(text.split())

def get_sentence_subdirectories(a_dir):
	return [name for name in listdir(a_dir)
		if isdir(join(a_dir, name)) and name.startswith('S')]

def collect_files(speaker_dir, max_files_per_speaker=None):
	"""Collect wav files for specific speakers.

	Returns:
	    list: List of collected wav files.
	"""
	paths = []
	if not isdir(speaker_dir):
		raise RuntimeError("{} doesn't exist.".format(speaker_dir))
	for sd in get_sentence_subdirectories(speaker_dir):
		files = [join(join(speaker_dir, sd), f) for f in listdir(join(speaker_dir, sd))]
		files = list(filter(lambda x: splitext(x)[1] == ".wav", files))
		files = sorted(files)
		files = files[:max_files_per_speaker]
		for f in files:
			paths.append(f)

	return paths

meta = [
	'_nurijimas', '_pilvas', '_pauze', '_tyla',
	'_ikvepimas', '_iskvepimas', '_garsas',
	'_puslapis', '_puslpais', '_kede', '_durys',
	'_cepsejimas'
]

meta_m = [
	('septyni_ty', 'septyni'), ('aštuoni_tuo', 'aštuoni'), ('devyni_vy','devyni'),
	('pirma_pir', 'pirma'), ('antra_an', 'antra'), ('trečia_tre', 'trečia'),
	('ketvirta_vir', 'ketvirta'), ('penkta_pen', 'penkta'), ('šešta_šeš', 'šešta'),
	('septinta_tin', 'septinta'), ('aštunta_tun', 'aštunta'), ('devinta_vin', 'devinta'),
	('dešimta_ši', 'dešimta'), ('procentų_cen', 'procentų'), ('vadinamaa_maa','vadinama'),
	('aplankų_ap', 'aplankų'), ('veiklų_veik', 'veiklų'), ('_įtrūkimu', 'įtrūkimu'),
	('sugriauta_ta', 'sugriauta'), ('laikomi_mi', 'laikomi'), ('siauros_siau', 'siauros'),
	('_padpadėtis', 'padpadėtis'), ('_klėstinčiu', 'klestinčiu'), ('langus_gus', 'langus'),
	('eštuoni_tuo', 'aštuoni'), ('architektūra_tū', 'architektūra'), ('rezultatus_ta', 'rezultatus'),
	('ketvyrta_vyr', 'ketvyrta'), ('_koplystulpiai', 'koplystulpiai')
]


def _process_utterance(mel_dir, linear_dir, wav_dir, index, wav_path, text, hparams):
	"""
	Preprocesses a single utterance wav/text pair

	this writes the mel scale spectogram to disk and return a tuple to write
	to the train.txt file

	Args:
		- mel_dir: the directory to write the mel spectograms into
		- linear_dir: the directory to write the linear spectrograms into
		- wav_dir: the directory to write the preprocessed wav into
		- index: the numeric index to use in the spectogram filename
		- wav_path: path to the audio file containing the speech input
		- text: text spoken in the input audio file
		- hparams: hyper parameters

	Returns:
		- A tuple: (audio_filename, mel_filename, linear_filename, time_steps, mel_frames, linear_frames, text)
	"""
	try:
		# Load the audio as numpy array
		wav = audio.load_wav(wav_path, sr=hparams.sample_rate)
	except FileNotFoundError: #catch missing wav exception
		print('file {} present in csv metadata is not present in wav folder. skipping!'.format(
			wav_path))
		return None

	#rescale wav
	if hparams.rescale:
		wav = wav / np.abs(wav).max() * hparams.rescaling_max

	#M-AILABS extra silence specific
	if hparams.trim_silence:
		wav = audio.trim_silence(wav, hparams)

	#Mu-law quantize
	if is_mulaw_quantize(hparams.input_type):
		#[0, quantize_channels)
		out = mulaw_quantize(wav, hparams.quantize_channels)

		#Trim silences
		start, end = audio.start_and_end_indices(out, hparams.silence_threshold)
		wav = wav[start: end]
		out = out[start: end]

		constant_values = mulaw_quantize(0, hparams.quantize_channels)
		out_dtype = np.int16

	elif is_mulaw(hparams.input_type):
		#[-1, 1]
		out = mulaw(wav, hparams.quantize_channels)
		constant_values = mulaw(0., hparams.quantize_channels)
		out_dtype = np.float32

	else:
		#[-1, 1]
		out = wav
		constant_values = 0.
		out_dtype = np.float32

	# Compute the mel scale spectrogram from the wav
	mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)
	mel_frames = mel_spectrogram.shape[1]

	if mel_frames > hparams.max_mel_frames and hparams.clip_mels_length:
		return None

	#Compute the linear scale spectrogram from the wav
	linear_spectrogram = audio.linearspectrogram(wav, hparams).astype(np.float32)
	linear_frames = linear_spectrogram.shape[1]

	#sanity check
	assert linear_frames == mel_frames

	#Ensure time resolution adjustement between audio and mel-spectrogram
	fft_size = hparams.n_fft if hparams.win_size is None else hparams.win_size
	l, r = audio.pad_lr(wav, fft_size, audio.get_hop_size(hparams))

	#Zero pad for quantized signal
	out = np.pad(out, (l, r), mode='constant', constant_values=constant_values)
	assert len(out) >= mel_frames * audio.get_hop_size(hparams)

	#time resolution adjustement
	#ensure length of raw audio is multiple of hop size so that we can use
	#transposed convolution to upsample
	out = out[:mel_frames * audio.get_hop_size(hparams)]
	assert len(out) % audio.get_hop_size(hparams) == 0
	time_steps = len(out)

	# Write the spectrogram and audio to disk
	audio_filename = 'speech-audio-{:05d}.npy'.format(index)
	mel_filename = 'speech-mel-{:05d}.npy'.format(index)
	linear_filename = 'speech-linear-{:05d}.npy'.format(index)
	np.save(os.path.join(wav_dir, audio_filename), out.astype(out_dtype), allow_pickle=False)
	np.save(os.path.join(mel_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)
	np.save(os.path.join(linear_dir, linear_filename), linear_spectrogram.T, allow_pickle=False)

	# Return a tuple describing this training example
	return (audio_filename, mel_filename, linear_filename, time_steps, mel_frames, text)
