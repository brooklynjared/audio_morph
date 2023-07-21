import librosa
import numpy as np
from scipy.io import wavfile

# Load audio files
y1, sr = librosa.load('78_Kick_SP_01.wav')
y2, sr = librosa.load('78_Kick_SP_09.wav')

# Extract pitch features
f0_1 = librosa.yin(y1, fmin=80, fmax=400)
f0_2 = librosa.yin(y2, fmin=80, fmax=400)


# Calculate length difference 
len_diff = len(f0_2) - len(f0_1)  

# Pad f0_1 with zeros
f0_1 = np.pad(f0_1, (0, len_diff))

# Average pitches
f0 = (f0_1 + f0_2) / 2

# Extract timbre features
spectro_1 = np.abs(librosa.stft(y1))
spectro_2 = np.abs(librosa.stft(y2))

# Average timbre
spectro = (spectro_1 + spectro_2) / 2

# Reconstruct audio
y = librosa.istft(spectro)

# Resynthesize pitch 
y_pitch = librosa.effects.pitch_shift(y, sr, n_steps=np.mean(f0-f0_1))

# Save output
wavfile.write('output.wav', sr, y_pitch)