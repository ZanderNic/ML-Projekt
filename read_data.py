import io
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
import librosa

db_file_name = "batcallsv14.db"
table_name = "batcalls"

data = pd.read_sql(table_name, f"sqlite:///{db_file_name}")
data = data.groupby('bat').head(300)
data['arr'] = data['arr'].map(lambda bytes: np.load(io.BytesIO(bytes)))
data['arr'] = data['arr'].map(lambda call: call.astype(np.float32) / 2.0**(16-1))

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

def fourier(data):
  stft = np.abs(librosa.stft(data, n_fft=512, hop_length=32))
  stft = 10 * np.log10(stft)
  stft = np.nan_to_num(stft)

  stft = (stft - np.min(stft)) / (np.max(stft) - np.min(stft))
  stft = np.reshape(stft, (257, 138, 1))
  stft = stft[:256, -128: , :]
  return stft

data['arr'] = data['arr'].map(lambda call: butter_bandpass_filter(call, 1500, 12000, 44100, 5))
data['arr'] = data['arr'].map(lambda call: (call - np.mean(call)) / np.std(call))
data['arr'] = data['arr'].map(lambda call: fourier(call))