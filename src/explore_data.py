#%% Imports
import scipy.io
from scipy import signal
from pathlib import Path 
from matplotlib import pyplot as plt
import numpy as np

#%% Read in data
datapath = Path('../data/')
datafile = 'P1_high1.mat'
mat = scipy.io.loadmat(datapath / datafile)


# %% Seperate data
fs = mat['fs'].squeeze()
trig = mat['trig']
eeg = mat['y']
cz=eeg[:,3]
cz = cz/np.max(cz)

print(f'Sampling Frequency {fs}\nInput signal shape {trig.shape}\nOutput signal shape {eeg.shape}')

# %% Bode plot of signal
def plot_freq(data, fs):
    freqs, psd = signal.welch( 
        x=data,
        fs=fs
    )
    bd = 10 * np.log10(psd)

    plt.figure()
    plt.subplot(121)
    plt.title('PSD')
    plt.semilogx(freqs, psd)
    plt.subplot(122)
    plt.title('Bode Plot')
    plt.semilogx(freqs, bd)

plot_freq(cz, fs)

# %% View signal
sample_window = 1000
trig_sample = trig[0:sample_window]
sample = cz[0:sample_window]
nsamples = sample.shape[0]
T = nsamples * 1/fs
t = np.linspace(0, T, nsamples, endpoint=False)

plt.figure()
plt.plot(t, sample, label='Raw data')
plt.plot(t, trig_sample, label='Trigger')
plt.legend(loc='best')

# %% Filtering 

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

def butter_highpass(lowcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = signal.butter(order, [low], btype='highpass')
    return b, a

def butter_highpass_filter(data, lowcut, fs, order=5):
    b, a = butter_highpass(lowcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

lc = 0.1
hc = 30

sos = signal.butter(10, lc, 'hp', fs=fs, output='sos')
filtered = signal.sosfilt(sos, cz)

plt.figure()
plt.plot(t, filtered[0:sample_window], label='Filtered signal')
plt.plot(t, sample, label='Unfiltered signal')
plt.plot(t, trig_sample, label='Trigger')
plt.legend(loc='best')
# %%

plot_freq(filtered, fs)


# %% Find the spacing of events
event_idxs = np.nonzero(trig)[0]

print(f'Number of events: {len(event_idxs)}')
print(f'Median number of samples between event: {np.median(np.diff(event_idxs))}')
print(f'Mean number of samples between event: {np.mean(np.diff(event_idxs))}')
print(f'Target count: {len(np.where(trig == 2)[0])}')
print(f'Non target count: {len(np.where(trig == 1)[0])}')
print(f'Distractor count: {len(np.where(trig == -1)[0])}')

# %% Split up the signal
window_size = 70
delay = 0
signal=eeg[:,3]
target_cnt, non_target_cnt, distractor_cnt = 0,0,0

for i, event_idx in enumerate(event_idxs):
    if trig[event_idx] == -1:
        distractor_cnt += 1
    elif trig[event_idx] == 1:
        non_target_cnt += 1
    elif trig[event_idx] == 2:
        target_cnt += 1







# %%
