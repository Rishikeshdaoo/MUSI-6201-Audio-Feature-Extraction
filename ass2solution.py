import os
import glob
import numpy as np
#import matplotlib

#matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt
from scipy.io.wavfile import read


eps = 1e-6


def hann(L):
    return 0.5 - (0.5 * np.cos(2 * np.pi / L * np.arange(L))).reshape(1, -1)


def fft(xb):
    K = len(xb[1])
    return np.abs(np.fft.rfft(xb*hann(K), K, axis=1))


def block_audio(x, blockSize, hopSize, fs):
    numBlocks = int(np.ceil(x.size / hopSize))
    xb = np.zeros([numBlocks, blockSize])
    t = (np.arange(numBlocks) * hopSize) / fs
    x = np.concatenate((x, np.zeros(blockSize)), axis=0)
    for n in range(numBlocks):
        i_start = n * hopSize
        i_stop = np.min([x.size - 1, i_start + blockSize - 1])
        xb[n][np.arange(0, blockSize)] = x[np.arange(i_start, i_stop + 1)]
    return xb, t


def extract_spectral_centroid(xb, fs):
    K = len(xb[1]) // 2
    X = fft(xb)
    v_sc = np.sum(np.arange(K + 1).reshape(1, -1) * X, axis=1) / (np.sum(X, axis=1) + eps) / (K - 1)
    return v_sc * fs / 2


def extract_rms(xb):
    return np.maximum(20 * np.log10(np.sqrt((1 / len(xb[1])) * np.sum(np.square(xb), axis=1)) + eps), -100)


def extract_zerocrossingrate(xb):
    return np.sum(np.abs(np.diff(np.sign(xb), axis=1)), axis=1) / (2 * len(xb[1]))


def extract_spectral_crest(xb):
    X = fft(xb)
    return np.max(X, axis=1) / np.sum(X, axis=1)


def extract_spectral_flux(xb):
    return np.insert(np.sqrt(np.sum(np.square(np.diff(fft(xb), axis=0)), axis=1)) / (len(xb[1]) / 2), 0, 0)


# A2
def extract_features(x, blockSize, hopSize, fs):
    xb, t = block_audio(x, blockSize, hopSize, fs)
    v_sc = extract_spectral_centroid(xb, fs)
    v_rms = extract_rms(xb)
    v_zc = extract_zerocrossingrate(xb)
    v_scf = extract_spectral_crest(xb)
    v_sf = extract_spectral_flux(xb)
    return np.vstack((v_sc, v_rms, v_zc, v_scf, v_sf))


label = {'sc_mean': 0, 'rms_mean': 2, 'zcr_mean': 4, 'scr_mean': 6, 'sf_mean': 8,
         'sc_std' : 1, 'rms_std': 3, 'zcr_std': 5, 'scr_std': 7, 'sf_std': 9}


# A3
def aggregate_feature_per_file(features):
    return np.dstack((np.mean(features, axis=1), np.std(features, axis=1))).flatten()


def wavread(path):
    sr, x = read(path)
    if x.dtype == 'float32':
        return sr, x
    elif x.dtype == 'uint8':
        return sr, (x / 128.) - 1
    else:
        bits = x.dtype.itemsize * 8
        return sr, x / (2 ** (bits - 1))


# A4
def get_feature_data(path, blockSize, hopSize):
    files = glob.glob(os.path.join(path, "*.wav"))
    ft_data = np.empty((10, len(files)))
    for i, f in enumerate(files):
        sr, audio = wavread(f)
        ft = extract_features(audio, blockSize, hopSize, sr)
        agg_ft = aggregate_feature_per_file(ft)
        ft_data[:, i] = agg_ft.flatten()
    return ft_data


# B1
def normalize_zscore(featureData):
    return ((featureData.T - np.mean(featureData, axis=1)) / np.std(featureData, axis=1)).T


def plot_features(features, index):
    plots = [['sc_mean', 'scr_mean'], ['sf_mean', 'zcr_mean'], ['rms_mean', 'rms_std'],
             ['zcr_std', 'scr_std'], ['sc_std', 'sf_std']]

    fig, ax = plt.subplots(3, 2, figsize=(12, 10))
    l1 = l2 = None
    for i, (x_axis, y_axis) in enumerate(plots):
        x = i // 2
        y = i % 2
        ax[x, y].set(title=x_axis + ' vs ' + y_axis, xlabel=x_axis, ylabel=y_axis, xlim=(-3, 3.5), ylim=(-3, 6), axisBelow=True)
        ax[x, y].grid(linestyle='dashed')
        l1 = ax[x, y].scatter(features[label[x_axis], :index], features[label[y_axis], :index], c='b')
        l2 = ax[x, y].scatter(features[label[x_axis], index:], features[label[y_axis], index:], c='r')
    ax[2, 1].remove()
    fig.legend([l1, l2], ["speech", "music"], loc=4)
    plt.tight_layout(pad=1)
    plt.show()


# C1
def visualize_features(path_to_musicspeech, blockSize=1024, hopSize=256):
    folders = ["speech_wav", "music_wav"]
    index = []
    ft_matrix = []
    for folder in folders:
        path = os.path.join(path_to_musicspeech, folder)
        ft_data = get_feature_data(path, blockSize, hopSize)
        ft_matrix.append(ft_data)
        index.append(ft_data.shape[1])
    ft_matrix = np.hstack((ft_matrix[0], ft_matrix[1]))
    norm_ft = normalize_zscore(ft_matrix)
    plot_features(norm_ft, index[0])


# visualize_features("music_speech")
