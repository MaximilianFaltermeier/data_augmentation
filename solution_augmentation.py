import random

import numpy as np
from numpy.random import randint

import os
from os.path import isfile, join
from shutil import rmtree
from tqdm import tqdm

from librosa import load, feature
from scipy import signal
from sklearn.mixture import GaussianMixture

import matplotlib.pyplot
import sounddevice
import soundfile as sf

SR = 22050


class GenSpeech:
    def __init__(self, path):
        self.path = path
        self.files = os.listdir(path)

    def __len__(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if len(self.files) == 0:
                raise StopIteration
                return
            file = self.files.pop()
            if isfile(join(self.path, file)):
                return sf.read(join(self.path, file))


def create_gen_noise(path, len_chunk=16000):
    files = os.listdir(path)
    while True:
        if len(files) == 0:
            raise StopIteration
            return
        ran_int = randint(0, len(files))
        off_set = 0
        increment = len_chunk / SR
        duration = increment
        while True:
            wave_file, _ = load(path + files[ran_int], sr=SR, duration=duration, offset=off_set)
            files.remove(files[ran_int])
            duration += increment
            off_set += increment
            if len(wave_file) < len_chunk - 1:
                # len_chunk-1 takes into consideration that depending on the sr and the len_chunk
                # the actual length can be len_chunk-1 due to the definition of the increment
                break
            yield wave_file


def create_gen_rirs(path):
    files = os.listdir(path)
    while True:
        if len(files) == 0:
            raise StopIteration
            return
        ran_int = randint(0, len(files))
        rir, _ = load(path + files[ran_int], sr=SR)
        files.remove(files[ran_int])
        yield rir


def mix(clean, noise, snr, rir, biquad):
    clean = signal.convolve(clean, rir)
    noise_rir = create_gen_rirs('./Noise/train').__next__()
    mixed = signal.convolve(noise, noise_rir) / snr + clean
    max_mixed = max(abs(mixed))
    if max_mixed > 1:
        mixed /= max_mixed
        clean /= max_mixed
    clean = signal.convolve(clean, biquad)
    mixed = signal.convolve(mixed, biquad)
    return clean, mixed


def feature_extraction(noisy):
    mfcc = feature.mfcc(noisy, sr=SR, n_mff=40, n_fft=512, window='hamming')
    delta1 = feature.delta(mfcc, sr=SR, order=1, width=12)
    delta2 = feature.delta(mfcc, sr=SR, order=2, width=6)
    zcr = feature.zero_crossing_rate(noisy)
    sc = feature.spectral_centroid(noisy, sr=SR)
    rolloff = feature.spectral_rolloff(noisy, sr=SR)
    bandwidth = feature.spectral_bandwidth(noisy, sr=SR)
    feature_matrix = np.stack([mfcc, delta1, delta2, zcr, sc, rolloff, bandwidth])
    return feature_matrix


def vad_extraction(clean):
    # todo: your code here
    return 0


def run_augmentation():
    path = './Noise/test'
    path_noise = './Noise/test'
    break_counter = 0
    for elem in GenSpeech(path=path):
        break_counter += 1
        if break_counter >= 10:
            break
        noise = create_gen_noise(path_noise, len_chunk=len(elem)).__next__()
        rir1 = create_gen_rirs('./RIR').__next__()
        rir2 = create_gen_rirs('./RIR').__next__()
        a1, b1, a2, b2 = [random.uniform(-3/8, 3/8) for _ in range(4)]
        biquad = signal.bilinear([b1, b2], [a1, a2])
        SNR = random.randint(0, 12)
        result_mix = mix(elem, noise, SNR, rir1, biquad)
        result_feat_ex = feature_extraction(result_mix[1])
        result_vad_ex = vad_extraction(elem)


if __name__ == '__main__':
    run_augmentation()
