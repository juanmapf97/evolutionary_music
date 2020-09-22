import os
import sys

import librosa

import tensorflow as tf
import numpy as np
import pandas as pd

from librosa.filters import mel
from scipy import signal
from scipy.fftpack import fft

import tensorflow.keras.backend as K

class OnsetDetectionCNN:
    def __init__(self):
        self._model = self._get_architecture()
    
    def _get_architecture(self):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(name='conv2d_1', input_shape=(80, 15, 3), filters=10, kernel_size=(3, 7), activation="relu"),
            tf.keras.layers.MaxPool2D(name='maxpool2d_1', pool_size=(3, 1)),
            tf.keras.layers.Conv2D(name='conv2d_2', filters=20, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPool2D(name='maxpool2d_2', pool_size=(3,1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, name='dense_1', activation="sigmoid"),
            tf.keras.layers.Dense(15, name='dense_2', activation="sigmoid"),
        ])
    
    def _make_frame(self, data, nhop, nfft):
        length = data.shape[0]
        framedata = np.concatenate((data, np.zeros(nfft)))  # zero padding
        return np.array([framedata[i*nhop:i*nhop+nfft] for i in range(length//nhop)])  

    def _preprocess_X(self, data_path):
        '''
        data: dataframe with audio paths
        '''
        sample_rate = 0
        paths_data = []
        data_paths = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
        print('Load audio data')
        for path in data_paths:
            data, sample_rate = librosa.load(os.path.join(data_path, path))
            paths_data.append(data)
        
        nhop=512
        '''
        These params are defined by the source paper in its experiment.
        '''
        nffts=[1024, 2048, 4096]
        mel_nband=80
        mel_freqlo=27.5
        mel_freqhi=16000.0

        paths_feat_channels = []
        print('Perform data transformation')
        for data in paths_data:
            feat_channels = []
            for nfft in nffts:
                feats = []
                window = signal.blackmanharris(nfft)
                filt = mel(sample_rate, nfft, mel_nband, mel_freqlo, mel_freqhi)
                
                # get normal frame
                frame = self._make_frame(data, nhop, nfft)
                # print(frame.shape)

                # melscaling
                processedframe = fft(window*frame)[:, :nfft//2+1]
                processedframe = np.dot(filt, np.transpose(np.abs(processedframe)**2))
                processedframe = 20*np.log10(processedframe+0.1)
                # print(processedframe.shape)

                feat_channels.append(processedframe)
            paths_feat_channels.append(np.array(feat_channels))
        
        return sample_rate, paths_feat_channels
    
    def _preprocess_y(self, data_path, preprocessed_X, sample_rate):
        '''
        data: path containing pandas DataFrames containing a timestamp for the onsets
        X_feature_shape: the resulting shape of the preprocessed X.
        '''
        preprocessed_ys = []
        data_paths = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
        print('Load onset data and convert.')
        for path, X in zip(data_paths, preprocessed_X):
            data = pd.read_csv(os.path.join(data_path, path))
            preprocessed_y = np.zeros(X.shape[2])
            
            '''
            Values in musicnet are miliseconds * (sample_rate/512), but the algorithm requires
            seconds * (sample_rate/512) so we convert it by just dividing it by 1000.
            '''
            timing = data['start_time'].values/1000
            events_index = np.rint(timing).astype(np.int32)
            events_index = np.delete(events_index, np.where(events_index >= X.shape[2]))

            preprocessed_y[events_index] = 1
            # milden
            for i in events_index:
                # Smooth to the left.
                if i > 0 and preprocessed_y[i-1] == 0:
                    preprocessed_y[i-1] = 0.25
                if i < len(preprocessed_y)-1 and preprocessed_y[i+1] == 0:
                    preprocessed_y[i+1] = 0.25
            preprocessed_ys.append(preprocessed_y)
        return np.array(preprocessed_ys)
    
    def _training_test_sets(self, X, y, train_split=0.7):
        train_xs = []
        train_ys = []
        test_xs = []
        test_ys = []
        # Extract fragments from each song
        for x, y_ in zip(X, y):
            idx = [i for i in range(0, x.shape[2]-15, 15)]
            rand_idx = np.random.choice(idx, size=len(idx), replace=False)
            split_idx = int(len(rand_idx) * train_split)

            train_idx = rand_idx[:split_idx]
            test_idx = rand_idx[split_idx:]

            # Training set
            train_x = []
            train_y = []
            for train_sample_idx in train_idx:
                train_sample_x = x[:, :, train_sample_idx:train_sample_idx+15]
                train_sample_y = y_[train_sample_idx:train_sample_idx+15]
                train_x.append(train_sample_x)
                train_y.append(train_sample_y)
            train_x = np.array(train_x)
            train_xs.append(train_x)
            train_y = np.array(train_y)
            train_ys.append(train_y)

            # Test set
            test_x = []
            test_y = []
            for test_sample_idx in test_idx:
                test_sample_x = x[:, :, test_sample_idx:test_sample_idx+15]
                test_sample_y = y_[test_sample_idx:test_sample_idx+15]
                test_x.append(test_sample_x)
                test_y.append(test_sample_y)
            test_x = np.array(test_x)
            test_xs.append(test_x)
            test_y = np.array(test_y)
            test_ys.append(test_y)
        
        train_X = np.concatenate(train_xs)
        train_X = np.swapaxes(train_X, 1, 2)
        train_X = np.swapaxes(train_X, 2, 3)
        train_y = np.concatenate(train_ys)
        test_X = np.concatenate(test_xs)
        test_X = np.swapaxes(test_X, 1, 2)
        test_X = np.swapaxes(test_X, 2, 3)
        test_y = np.concatenate(test_ys)

        return train_X, train_y, test_X, test_y

    def step_decay(self, epoch):
        initial_lrate = 0.05
        if epoch >= 10:
            lrate = initial_lrate + (0.9 * min(10, epoch-10))
        else:
            lrate = initial_lrate
        return lrate
    
    def custom_loss(self, y_true, y_pred):
        return K.mean(K.square(y_pred - y_true), axis=-1)

    def train(self, raw_train_X, raw_train_y):
        '''
        raw_train_X: folder path with audio data
        raw_train_y: folder path with csvs containing onset times
        '''
        sample_rate, preprocessed_X = self._preprocess_X(raw_train_X)
        preprocessed_y = self._preprocess_y(raw_train_y, preprocessed_X, sample_rate)
        trainX, trainY, testX, testY = self._training_test_sets(preprocessed_X, preprocessed_y)

        optimizer = tf.keras.optimizers.SGD(
            learning_rate=0.05, momentum=0.45, nesterov=False, name="SGD"
        )
        lrate = tf.keras.callbacks.LearningRateScheduler(self.step_decay)
        callbacks_list = [lrate]

        self._model.compile(optimizer=optimizer, loss="mse", metrics=["mse"])
        self._model.summary()
        self._model.fit(trainX, trainY, epochs=100, batch_size=256, callbacks=callbacks_list)
        print(self._model.predict(testX)[0])
        print(trainY[0])

if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise Exception('Please specify the train_data and train_labels folders')
    odcnn_model = OnsetDetectionCNN()
    odcnn_model.train(sys.argv[1], sys.argv[2])