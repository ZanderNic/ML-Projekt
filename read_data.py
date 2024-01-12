import io
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
import librosa

def read_data(nrows = 300, db_file_name = './batcallsv14.db', table_name = 'batcalls'):
  data = pd.read_sql(table_name, f"sqlite:///{db_file_name}")
  data = data.groupby('bat').head(nrows)
  data['arr'] = data['arr'].map(transform_arr)
  return data


def transform_arr(bytes):
  data = np.load(io.BytesIO(bytes))
  data = data.astype(np.float32) / 2.0 ** (16-1)
  data = butter_bandpass_filter(data, 1500, 12000, 44100, 5)
  data = (data - np.mean(data)) / np.std(data)
  data = fft(data)
  return data


def butter_bandpass(lowcut, highcut, fs, order=5):
  nyq = 0.5 * fs
  low = lowcut / nyq
  high = highcut / nyq
  b, a = butter(order, [low, high], btype='band')
  return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
  b, a = butter_bandpass(lowcut, highcut, fs, order=order)
  y = lfilter(b, a, data)
  return y

def fft(data):
  # these are probably cargo-culted
  stft = np.abs(librosa.stft(data, n_fft=512, hop_length=32))
  stft = 10 * np.log10(stft)
  stft = np.nan_to_num(stft)

  stft = np.reshape(stft, (257, 138, 1))

  # BEWARE: I removed some data here, this is different from the original (old) version!
  # it seems like very high/low frequencies don't contain any useful data - 
  # this improves the performance of low-dimensional PCA by roughly 5%
  stft = stft[16:180, -128: , :]
  # scale to [0,1]
  stft = (stft - np.min(stft)) / (np.max(stft) - np.min(stft))
  return stft
