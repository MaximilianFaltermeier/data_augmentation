import random

import numpy as np
from numpy.random import randint
import math
import os
from os.path import isfile, join
from shutil import rmtree
from tqdm import tqdm

from librosa import load, feature
from scipy import signal
from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as pt
import sounddevice as sd
import soundfile as sf

SR = 16000


class GenSpeech:
    def __init__(self, path, split=None):
        self.path = path
        self.files = os.listdir(path)
        self.wave_files = []
        for file in self.files:
            self.wave_files += [(os.listdir(join(self.path, file)), join(self.path, file))]

    def __len__(self):
        return len(self.wave_files)

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if len(self.wave_files) == 0:
                raise StopIteration
                return
            for elem in self.wave_files:
                if elem[0] is []:
                    self.wave_files.remove(elem)
                    continue
                file = elem[0].pop()
                return load(join(elem[1], file), sr=SR)


def create_gen_noise(path, len_chunk=16000, split='train'):
    path = join(path, split)
    files = os.listdir(path)
    while True:
        if len(files) == 0:
            raise StopIteration
            return
        ran_int = randint(0, len(files))
        off_set = 0
        increment = len_chunk / SR
        duration = increment
        wave_file = []
        while len(wave_file) < len_chunk - 1:
            # len_chunk-1 takes into consideration that depending on the sr and the len_chunk
            # the actual length can be len_chunk-1 due to the definition of the increment
            wave_file, _ = load(join(path, files[ran_int]), sr=SR, duration=duration, offset=off_set)
            duration += increment
            off_set += increment
            yield wave_file


def create_gen_rirs(path, split='train'):
    files = os.listdir(path)
    # files_len = len(files)
    # if split == 'train':
    while True:
        if len(files) == 0:
            raise StopIteration
            return
        ran_int = randint(0, len(files))
        rir, _ = load(join(path, files[ran_int]), sr=SR)
        yield rir


def mix(clean, noise, snr, rir, biquad):
    new_clean = signal.fftconvolve(clean[0], rir, mode='same')
    noise_rir = next(create_gen_rirs('./RIR'))
    mixed = signal.fftconvolve(noise, noise_rir, mode='same')
    a = sum(new_clean**2)/(10 ** (snr / 20))/sum(mixed**2)
    mixed = mixed*a + new_clean
    max_mixed = max(abs(mixed))
    if max_mixed > 1:
        mixed /= max_mixed
        new_clean /= max_mixed
    new_clean = signal.sosfilt(biquad, new_clean)
    mixed = signal.sosfilt(biquad, mixed)
    return new_clean, mixed


def feature_extraction(noisy):
    if any(np.isinf(noisy)):
        print('noisy is nan')
    mfcc = feature.mfcc(noisy, sr=SR, n_mfcc=40, n_fft=512, window='hamming', win_length=320, hop_length=160)
    delta1 = feature.delta(mfcc, order=1)[:12]
    delta2 = feature.delta(mfcc, order=2)[:6]
    zcr = feature.zero_crossing_rate(noisy, hop_length=160)
    sc = feature.spectral_centroid(noisy, sr=SR, hop_length=160)
    rolloff = feature.spectral_rolloff(noisy, sr=SR, hop_length=160)
    bandwidth = feature.spectral_bandwidth(noisy, sr=SR, hop_length=160)
    feature_matrix = np.vstack([mfcc, delta1, delta2, zcr, sc, rolloff, bandwidth])
    return feature_matrix


def vad_extraction(clean):
    log_power = np.log10(feature.rms(clean, frame_length=320, hop_length=160))
    model = GaussianMixture(n_components=2)
    model.fit(np.reshape(log_power, (-1, 1)))
    mean_min, mean_max = min(model.means_), max(model.means_)
    if abs(mean_min) > abs(mean_max):
        index = list(model.means_).index(mean_min)
    else:
        index = list(model.means_).index(mean_max)
    post_prob = model.predict_proba(np.reshape(log_power, (-1, 1)))
    return post_prob[:, index]


def run_augmentation():
    path = './speech'
    path_noise = './Noise'
    break_counter = 0
    for elem in GenSpeech(path=path):
        # limit amount of speech files
        break_counter += 1
        if break_counter >= 20:
            break

        # get rir
        rir1 = next(create_gen_rirs('./RIR'))
        rir2 = next(create_gen_rirs('./RIR'))

        # definition of biquad filter
        a1, b1, a2, b2 = [random.uniform(-3 / 8, 3 / 8) for _ in range(4)]
        biquad =[1, b1, b2, 1, a1, a2]

        # generate random SNR
        SNR = random.randint(0, 12)

        # get noise
        noise = []
        while len(noise) < len(elem[0]):
            noise.extend(next(create_gen_noise(path_noise, len_chunk=16000)))

        # call functions
        result_mix = mix(elem, noise[:len(elem[0])], SNR, rir1, biquad)
        result_feat_ex = feature_extraction(result_mix[1])
        result_vad_ex = vad_extraction(elem[0])
        pt.figure(1)
        pt.plot(result_mix[1])
        pt.plot(result_mix[0])
        #pt.plot(range(len(result_mix[1])), result_vad_ex)
        pt.title('posterior probabilities')
        pt.show()
        np.savez('results_file', result_mix, result_feat_ex, result_vad_ex)
        print('___')


if __name__ == '__main__':
    run_augmentation()
    print('end')