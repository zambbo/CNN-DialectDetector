import librosa
import librosa.display
import matplotlib.pyplot as plt
import json
import numpy as np
import pickle
import argparse
import os
from config import Config
import sys
import multiprocessing as mp



class Preprocessor:

    def __init__(self, config):
        self.config = config
        pass
    
    def resize(self, audio, target_size):
        if len(audio) > target_size: data = audio[:target_size]
        else: data = librosa.util.pad_center(audio, size=target_size)
        return data

    def chromagram(self, audio, samplerate, n_fft, hop_length):
        chroma = librosa.feature.chroma_stft(audio, samplerate, n_fft=n_fft, hop_length=hop_length)
        return chroma
    
    def mfcc(self, audio, samplerate, n_fft, hop_length):
        mfcc = librosa.feature.mfcc(audio, samplerate, n_mfcc=100, n_fft= n_fft, hop_length= hop_length)
        return mfcc

    def spectrogram(self, audio, samplerate, n_fft, hop_length):
        stft = librosa.stft(audio, n_fft= n_fft, hop_length= hop_length)
        magnitude = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(magnitude)
        return log_spectrogram

    def transform(self, audios, mode):
        if mode == "chroma":
            t_audios = [self.chromagram(audio,self.config.samplerate, self.config.n_fft, self.config.hop_length) for audio in audios]
        elif mode == "mfcc":
            t_audios = [self.mfcc(audio,self.config.samplerate, self.config.n_fft, self.config.hop_length) for audio in audios]
        elif mode == "spectro":
            t_audios = [self.spectrogram(audio,self.config.samplerate, self.config.n_fft, self.config.hop_length) for audio in audios]

        t_audios = np.stack(t_audios, axis=0)
        return t_audios

    def saveFig(self, data, fig_path, mode):
        if mode=='chroma':  
            librosa.display.specshow(data, x_axis='time', y_axis=mode)
            plt.savefig(fig_path)
            plt.title("Chromagram")
            plt.close()
        elif mode=='spectro':
            librosa.display.specshow(data, sr= self.config.samplerate, hop_length=self.config.hop_length)
            plt.xlabel("Time")
            plt.ylabel("Frequency")
            plt.colorbar(format="%+2.0f dB")
            plt.title("Spectrogram (dB)")
            plt.savefig(fig_path)
            plt.close()
        elif mode=='mfcc':
            librosa.display.specshow(data, x_axis='time')
            plt.savefig(fig_path)
            plt.title("MFCC")
            plt.close()

    def savePickle(self, data, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(data, f)

    def loadJson(self, json_path):
        with open(json_path, "r") as f:
            return json.load(f) 

    def parsing_time_info(self, label_path):
        json_data = self.loadJson(label_path)
        dialog = json_data['utterance']
        time_info = [(d['start'], d['end']) for d in dialog]
        
        return time_info

    def audio_preprocess(self, label_path, data):
        time_info = self.parsing_time_info(label_path)

        audio, sr = data
        try:
            splited_audios = [audio[int(start*sr):int(end*sr)] for start, end in time_info]
            splited_audios = [audio for audio in splited_audios if len(audio) >= self.config.drop_start_sec*self.config.samplerate and len(audio) <= self.config.drop_end_sec*self.config.samplerate]
            splited_audios = [self.resize(audio, self.config.resize) for audio in splited_audios]
            padded_audios = np.vstack(splited_audios)
        except: raise Exception("Exception while audio preprocessing...")

        return padded_audios

    def preprocessing(self, label_path, data_path, file_name):
        if len(self.config.modes) == 0:
            print("no mode")
            sys.exit(0)
        data = librosa.load(data_path, sr=16000)

        try:
            padded_audios = self.audio_preprocess(label_path, data)
        except Exception as e:
            print(e)
            return

        save_path = os.path.join(self.config.save_region_dir, file_name)
        if not os.path.isdir(save_path): os.mkdir(save_path)

        for mode in self.config.modes:
            preprocessed_audios = self.transform(padded_audios, mode)
            if self.config.img_save:
                save_fig_path = os.path.join(save_path, f"{file_name}_{mode}.png")
                self.saveFig(preprocessed_audios[0], save_fig_path, mode) # 첫번째만 출력
            save_pickle_path = os.path.join(save_path, f"{file_name}_{mode}.pickle")
            self.savePickle(preprocessed_audios, save_pickle_path)  

    def run(self):

        json_file_names = os.listdir(self.config.label_dir)
        json_file_names = [j for j in json_file_names if j.endswith('.json')]

        wav_file_name_dicts = dict()
        for data_dir_name, data_dir in zip(self.config.data_dir_names, self.config.data_dirs):
            wav_file_name_dicts[data_dir_name] = os.listdir(data_dir)
            wav_file_name_dicts[data_dir_name] = [w for w in wav_file_name_dicts[data_dir_name] if w.endswith('.wav')]

        wav_label_dict = dict()
        for json_file_name in json_file_names:
            for wav_dir_name, wav_file_names in wav_file_name_dicts.items():
                matched_wav_file = list(filter(lambda x: x[:-3] == json_file_name[:-4], wav_file_names))
                if len(matched_wav_file) != 0:
                    matched_wav_file[0] = (matched_wav_file[0], wav_dir_name)
                    break
            
            if len(matched_wav_file) == 0: continue
            
            wav_label_dict[json_file_name] = matched_wav_file[0]
        

        if not os.path.isdir(self.config.save_region_dir): os.mkdir(self.config.save_region_dir)

        cores = mp.cpu_count()
        pool = mp.Pool(cores)
        pool.map(self.p_run, list(wav_label_dict.items()))
        pool.close()
        pool.join()
        
    def p_run(self, data):
        label_file, data_file = data

        data_file_name, data_dir_name = data_file
        label_path = os.path.join(self.config.label_dir, label_file)
        data_path = os.path.join(self.config.region_dir, data_dir_name, data_file_name)
        file_name = data_file_name[:-4]
        print(file_name)
        self.preprocessing(label_path, data_path, file_name)
        return None

def argparsing():
    parser = argparse.ArgumentParser(description='Parser for preprocess korean dialect audio data')

    parser.add_argument('--chromagram', '-C', action='store_true', default=False,
    help="Preprocess data using chromagram")
    parser.add_argument('--spectrogram', '-S', action='store_true', default=False,
    help="Preprocess data using spectrogram")
    parser.add_argument('--mfcc', '-M', action='store_true', default=False,
    help="Preprocess data using MFCC")
    parser.add_argument('--figure', '-F', action="store", type=bool, default=False,
    help="Save figure if you set True")

    args = parser.parse_args()

    return args        

def main():
    args = argparsing()
    modes = []

    vis=False
    if args.figure: vis=True

    if args.mfcc:  modes.append('mfcc')
    if args.spectrogram: modes.append("spectro")
    if args.chromagram: modes.append("chroma")
    
    # 1. 데이터를 담고있는 전체 디렉토리
    # 2. 데이터 디렉토리 안의 지역 디렉토리
    # 3. 지역 디렉토리 안의 레이블 디렉토리(1개만 존재)
    # 4. 지역 디렉토리 안의 데이터 디렉토리 리스트 ex) ['gangwon_data_1','gangwon_data_2']
    # 5. 전처리한 데이터가 담길 베이스 디렉토리
    # 6. Samplerate (보통 16000으로 고정)
    # 7. drop_start_sec
    # 8. drop_end_sec
    # 9. img_save : True면 이미지 저장
    # 10. modes: 적용할 transform 리스트
    config = Config('../dataset', 'jeju', 'jeju_label', ['jeju_data_1'], './multi_test', 16000, 4, 6, vis, modes)

    preprocessor = Preprocessor(config)

    preprocessor.run()

if __name__ == '__main__':
    main()

    

