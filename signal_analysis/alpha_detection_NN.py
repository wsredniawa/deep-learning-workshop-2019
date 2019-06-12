# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 15:14:47 2018

@author: Wladek
"""

import tensorflow as tf
import pylab as py
import numpy as np
from neo.io import Spike2IO
from scipy.signal import spectrogram,filtfilt, butter

def plot_signal():
    '''rysowanie sygnalu'''
    py.figure()
    x_axis = np.linspace(0, len(eeg_signals[ch_name])/Fs, len(eeg_signals[ch_name]))
    py.title(ch_name)
    py.plot(x_axis, eeg_signals[ch_name])
    for time in events:
        py.axvline(time, color='r',linestyle = '--')
#    py.show()

def wczytywanie_sygnalu():
    r = Spike2IO(filename=fname)
    bl = r.read()[0]
    eeg_signals = {}
    events = []
    for seg in bl.segments:
        for asig in seg.analogsignals:
            print('nazwa kanalu: ', asig.name[-9:-7])
            eeg_signals[asig.name[-9:-7]] = asig
        for st in seg.events[2]:
            events.append(float(st))
    channel_names = list(eeg_signals.keys())
    Fs = int(eeg_signals['p1'].sampling_rate)
    return eeg_signals, events[2:], channel_names, Fs

mainDir = '/Users/Wladek/Dysk Google/Neuro_cwicz_komp/dane/EEG_2019/'
fname = mainDir + 'alfa.smr'
ch_name = 'O1'
eeg_signals, events, channel_names, Fs = wczytywanie_sygnalu()
plot_signal()
#%%
signal =  eeg_signals[ch_name]
'''prepare data set'''
examples = 14000
x_set = np.zeros((examples, Fs))
for i in range(7):
    ev_len = events[2*i+1] - events[2*i]
    tmpts = np.random.rand(1000)*(ev_len-1)
    for n in range(1000):
        x_set[i*1000+n] = signal[int(events[2*i]+tmpts[n])*Fs:int(events[2*i]+tmpts[n]+1)*Fs].T
tmpts = np.random.rand(7000)*(events[0]-100)
for i in range(7000):
    x_set[7000+i]= signal[int(tmpts[i])*Fs:int(tmpts[i]+1)*Fs].T

train_n = 3500
y_train = np.array(list(np.ones(train_n))+list(np.zeros(train_n)))
x_train = np.concatenate((x_set[:train_n], x_set[7000:7000+train_n]))
'''build and train model'''
y_test = y_train
x_test = np.concatenate((x_set[3500:7000], x_set[10500:]))

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(20, activation=tf.nn.relu),
  tf.keras.layers.Dense(20, activation=tf.nn.relu),
  tf.keras.layers.Dense(20, activation=tf.nn.relu),
  tf.keras.layers.Dense(2, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100)
st = model.evaluate(x_test, y_test)
#%%
y_new = []
y = np.zeros((1, 500))
for i in range(int(len(signal)/Fs)):
    y[0] = signal[i*Fs:(i+1)*Fs].T
    y_new.append(model.predict(y)[0][1])
#%%
py.figure()
shift = 0
y_new = np.array(y_new)
[b,a] = butter(3.,[0.1/(Fs/2.0), 200/(Fs/2.0)],btype = 'bandpass')
signal_filt = filtfilt(b, a, signal.T)
freq,time,Spec = spectrogram(signal_filt[0], Fs)
py.pcolormesh(time, freq, Spec, vmax = 10)
for time in events:
    py.axvline(time, color='r',linestyle = '--')
py.ylabel('Frequency [Hz]', fontsize=20)
py.xlabel('Time [sec]', fontsize=20)
py.xticks(fontsize=20)
py.yticks(fontsize=20)
py.ylim(-5, 75)
y_values = y_new*(-3)+shift
x_values = np.linspace(0, y_new.shape[0], y_new.shape[0])
py.plot(x_values, y_values, color = 'green')
py.fill_between(x_values, y_values, np.zeros(y_new.shape[0])+shift, color = 'y', alpha = 0.5)