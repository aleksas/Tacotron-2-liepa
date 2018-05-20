from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os
from preprocessor import _process_utterance
from os.path import join

import codecs
import chardet

def build_from_path(hparams, input_dirs, speaker, max_files_per_speaker, mel_dir, linear_dir, wav_dir, n_jobs=12, tqdm=lambda x: x):
	"""
	Preprocesses the speech dataset from a gven input path to given output directories

	Args:
		- hparams: hyper parameters
		- input_dirs: input directories that contain the files to prerocess
		- speaker: speaker iu (subfolder)
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
		with open(os.path.join(input_dir, 'metadata.csv'), encoding='utf-8') as f:
			for wav_path in collect_files(input_dir, speaker, max_files_per_speaker):
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

    with codecs.open(txt_path, encoding=charenc) as fin:
        text = fin.read()
        for m in meta:
            text = text.replace(m, '')
        for mm_p, mm_r in meta_m:
            text = text.replace(mm_p, mm_r)
        text = text.replace('\n', ' ').replace('\r', '').strip()

def get_sentence_subdirectories(a_dir):
    return [name for name in listdir(a_dir)
            if isdir(join(a_dir, name)) and name.startswith('S')]

def collect_files(data_root, speaker, max_files_per_speaker=None):
    """Collect wav files for specific speakers.

    Returns:
        list: List of collected wav files.
    """
    speaker_dir = join(data_root, speaker)
    paths = []

    for (i, d) in enumerate([speaker_dir]):
        if not isdir(d):
            raise RuntimeError("{} doesn't exist.".format(d))
        for sd in get_sentence_subdirectories(d):
            files = [join(join(speaker_dir, sd), f) for f in listdir(join(d, sd))]
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
    ('_padpadėtis', 'padpadėtis'), ('_klėstinčiu', 'klėstinčiu'), ('langus_gus', 'langus'),
    ('eštuoni_tuo', 'aštuoni'), ('architektūra_tū', 'architektūra'), ('rezultatus_ta', 'rezultatus'),
    ('ketvyrta_vyr', 'ketvyrta'), ('_koplystulpiai', 'koplystulpiai')
]
