import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.signal as sig
import mne
from scipy.fft import fft, fftfreq
from my_round import my_round
from baseline_removal import baseline_removal

# Set parameters
fs = 250 # Sample rate [Hz]
shift_time = 0.5 # Eye movement time [s]
delay_time = 0.13 # Reaction time [s]
end_time = 5 # End time [s]
channel_locs = np.array([48,54,55,56,57,58,61,62,63]) # 9 channels[Pz,PO5,PO3,POz,PO4,PO6,O1,Oz,O2]
target = 30 # Target number
channel = 7 # Channel number
topo_target = 30 # Topomap target number
Wp = np.array([7,70])/(fs/2)
Ws = np.array([6,78])/(fs/2)
channel_info_path = '../Data/BenchmarkDataset/64-channels.loc'
channel_names = ['FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8','FT7','FC5',
                 'FC3','FC1','FCz','FC2','FC4','FC6','FT8','T7','C5','C3','C1','Cz','C2','C4','C6','T8',
                 'M1','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8','M2','P7','P5','P3','P1','PZ',
                 'P2','P4','P6','P8','PO7','PO5','PO3','POz','PO4','PO6','PO8','CB1','O1','Oz','O2','CB2']

# Open a EEG data file
raweeg = sio.loadmat('../Data/BenchmarkDataset/S2.mat')['data']
raweeg = np.transpose(raweeg,[2,0,1,3])
eeg = raweeg[:,channel_locs,:,:]

# Baseline Removal
shift_len = my_round(shift_time*fs) # Eye movement time [samples]
shift_points = np.arange(shift_len) # Eye movement sample points [samples]
eeg = baseline_removal(eeg, shift_points) # Remove baseline

# Acquire Valid Data
offset_len = my_round((delay_time+shift_time)*fs)
end_len = my_round(end_time*fs)
eeg = eeg[:,:,offset_len:end_len,:]

# Choose a target
eeg = np.squeeze(eeg[target,channel,:,:])
eeg = np.squeeze(np.mean(eeg, axis = 1))
O, W = sig.cheb1ord(Wp,Ws,3,40)
B, A = sig.cheby1(O,0.5,W,'bandpass')
w, h = sig.freqz(B,A,1000,fs=fs,whole=False)
plt.plot(w, 20 * np.log10(abs(h)))
plt.title('Chebyshev Type I bandpass frequency response')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [dB]')
plt.show()
eeg = sig.filtfilt(B,A,eeg)
time_vec = np.linspace(0, eeg.shape[0]/fs, eeg.shape[0])
plt.plot(time_vec, eeg)
plt.title('Filtered EEG')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [V]')
plt.show()

# Fourier
N = eeg.shape[0]
eeg_f = fft(eeg)[:N//2]
freq_vec = fftfreq(N,1/fs)[:N//2]
plt.plot(freq_vec,2.0/N*np.abs(eeg_f))
plt.title('Frequency Spectrum of EEG')
plt.xlabel('Frequency [Hz]')
plt.show()

# Electrical Activity Mapping
avg_eeg = np.squeeze(raweeg[topo_target,:,:,:])
avg_eeg = np.squeeze(np.mean(avg_eeg,axis=2)) # Average over trials
info = mne.create_info(channel_names, sfreq=fs, ch_types='eeg')
evoked = mne.EvokedArray(avg_eeg,info,baseline = (0,0.5))
montage = mne.channels.read_custom_montage(channel_info_path)
evoked.set_montage(montage)
times = np.arange(4,5,0.2)
evoked.plot_topomap(times, ch_type="eeg", image_interp="cubic")
plt.show()