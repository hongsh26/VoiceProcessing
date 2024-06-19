import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)
def save_spectrogram(path):
    audio, sample_rate = librosa.load(path)
    # D = librosa.stft(audio)
    n_fft = 2048  # 창 크기
    hop_length = n_fft//4  # 홉 크기
    S = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=128)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    dir_path = os.getenv('dir_path')

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file_name = os.path.basename(path).replace('.flac', '.png')
    save_path = os.path.join(dir_path, file_name)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sample_rate, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.savefig(save_path)
    plt.close()
# file_path = '/Users/hongseunghyuk/PycharmProjects/deepVoiceTest/data/train_data_01/003/106/106_003_0077.flac'
# audio, sample_rate = librosa.load(file_path)
#
# # FT -> Spectrum
# fft = np.fft.fft(audio)
# print("fft")
# print(fft)
# fft = abs(fft)
# f = np.linspace(0, sample_rate, len(fft))
# left_spectrum = fft[:int(len(fft) / 2)]
# left_f = f[:int(len(fft) / 2)]
#
# #STFT -> Spectrogram
#
# plt.figure()
# librosa.display.waveshow(audio, sr=sample_rate, alpha=0.5)
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# plt.title("Waveform")
# S = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128)
# S_db = librosa.amplitude_to_db(S, ref=np.max)
# plt.figure()
# librosa.display.specshow(S_db, sr=sample_rate, x_axis='time', y_axis='mel')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Mel Spectrogram')
# plt.show()
data_path = os.getenv('data_path')
directory = data_path
list = os.listdir(directory)
flac_files = [file for file in list if file.endswith('.flac')]
for file_name in flac_files:
    print(file_name)
    save_spectrogram(directory + file_name)