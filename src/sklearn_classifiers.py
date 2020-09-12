#%% Imports
import scipy.io
from scipy import signal
from scipy.integrate import simps
from pathlib import Path 
from matplotlib import pyplot as plt
import numpy as np
import os
import glob
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier



#%% Read in data
datapath = Path('../data/')
files = list(datapath.glob('*high*.mat'))

file = files[0]
mat = scipy.io.loadmat(file)

# Seperate data
fs = mat['fs'].squeeze()
trig = mat['trig']
eeg = mat['y']


# %% Find the spacing of events
event_idxs = np.nonzero(trig)[0]
num_events = len(event_idxs)

print(f'Number of events: {num_events}')
print(f'Median number of samples between event: {np.median(np.diff(event_idxs))}')
print(f'Mean number of samples between event: {np.mean(np.diff(event_idxs))}')
print(f'Target count: {len(np.where(trig == 2)[0])}')
print(f'Non target count: {len(np.where(trig == 1)[0])}')
print(f'Distractor count: {len(np.where(trig == -1)[0])}')


# %% Split up the signal
window_size = 150

targets = np.empty((window_size, 8, 0))
non_targets = np.empty((window_size, 8, 0))

for event_idx in event_idxs:
    if trig[event_idx] == 1:
        non_targets = np.append(non_targets, np.expand_dims(eeg[event_idx:event_idx + window_size], axis=2), axis = 2)
    elif trig[event_idx] == 2:
        targets = np.append(targets, np.expand_dims(eeg[event_idx:event_idx + window_size], axis=2), axis = 2)


# %% Inspect data

sample = targets[:,:,0]
nsamples = sample.shape[0]
T = nsamples * 1/fs
t = np.linspace(0, T, nsamples, endpoint=False)

plt.figure()
plt.plot(t, sample, label='Target raw data')
# %% Singularity functions

def singularity(data, func):
    """Reduce a time series to a single value using some function"""
    return func(data)

def powerband(data, fs=256, lc=0.1, hc=30):
    """Compute power of a signal in a specified freq band"""
    freqs, psd = signal.welch(data, fs)
    idx_delta = np.logical_and(freqs >= lc, freqs <= hc)
    freq_res = freqs[1]
    return simps(psd[idx_delta], dx=freq_res)

# %% Calculate the power in relevant freq band
tpbs = np.empty((60, 8))
for event in range(60):
    for channel in range(8):
        tpbs[event, channel] = singularity(targets[:,channel,event], powerband)


ntpbs = np.empty((60, 8))
for event in range(60):
    for channel in range(8):
        ntpbs[event, channel] = singularity(non_targets[:,channel,event], powerband)


# %% Create SK-ready data
X = np.concatenate((tpbs, ntpbs), axis=0)
Y = np.concatenate((
    np.ones((tpbs.shape[0], 1)), 
    np.zeros((ntpbs.shape[0], 1))), 
    axis=0
)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

# %% LDA 
# - P1_high2.mat - Accuracy: 0.54

model = LinearDiscriminantAnalysis()
model.fit(x_train, y_train)

# %% KNN
# - P1_high2.mat - Accuracy: 0.708 (best accuracy ranging from num_nieghbors [1-9])

n_neighbors=5
weights = 'uniform'
model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
model.fit(x_train, y_train)

# %% SVM
# - P1_high2.mat - Accuracy (kernel=poly): 0.83
#                - Accuracy (kernel=linear/rbf): 0.75

model = SVC(gamma='scale', kernel='rbf')
model.fit(X,Y)

# %% Random Forest Classifier


# %% Predict and plot confusion matrix

predictions = model.predict(x_test)
score = model.score(x_test, y_test)
cm = metrics.confusion_matrix(y_test, predictions)

# Plot confusion matrix
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.title(f'Accuracy Score: {score}', size = 15)
# %%
