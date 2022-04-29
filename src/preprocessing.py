import librosa
import librosa.display
import matplotlib.pyplot as plt
import json
import numpy as np
import pickle
import argparse
import os

class Preprocessor:

    def __init__(self):
        pass
    
    def resize(self, audio, target_size):
        if len(audio) > target_size: data = audio[:target_size]
        else: data = librosa.util.pad_center(audio, size=target_size)
        return data

    def chromagram(self, audio, samplerate):
        chroma = librosa.feature.chroma_stft(audio, samplerate, n_fft=int(samplerate/40), hop_length=int(samplerate/100))
        return chroma
    
    def mfcc(self):
        pass

    def spectrogram(self):
        pass

    def saveChromaFig(self, chroma, fig_path):
        librosa.display.specshow(chroma, x_axis='time', y_axis='chroma')
        plt.savefig(fig_path)

    def saveSpectroFig(self, spectrogram, fig_path):
        librosa.display.specshow(spectrogram, x_axis='time', y_axis='spectro')
        plt.savefig(fig_path)

    def saveMFCCFig(self, mfcc, fig_path):
        librosa.display.specshow(mfcc, x_axis='time', y_axis='mfcc')
        plt.savefig(fig_path)    

    def saveChromaAsPickle(self, chroma, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(chroma, f)

    def saveMFCCAsPickle(self, mfcc, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(mfcc, f)    

    def saveSpectroAsPickle(self, spectro, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(spectro, f)        


def loadJson(json_path):
    with open(json_path, "r") as f:
        return json.load(f) 

def getDurations(json_s):
    durations = [(j['start'], j['end']) for j in json_s['utterance']]
    return durations
    
def split_audio(y, sr, durations):
    chunks = [y[int(start*sr):int(end*sr)] for start, end in durations]
    return chunks

def argparsing():
    parser = argparse.ArgumentParser(description='Parser for preprocess korean dialect audio data')

    parser.add_argument('--chromagram', '-C', action='store_true', default=False,
    help="Preprocess data using chromagram")
    parser.add_argument('--spectrogram', '-S', action='store_true', default=False,
    help="Preprocess data using spectrogram")
    parser.add_argument('--mfcc', '-M', action='store_true', default=False,
    help="Preprocess data using MFCC")

    args = parser.parse_args()

    return args

def main(base_dir, label_dir, data_dir, save_region_dir):

    data_dir = os.path.join(base_dir, data_dir)
    label_dir = os.path.join(base_dir, label_dir)

    args = argparsing()

    if args.mfcc:  print("mfcc")
    if args.spectrogram: print("spectrogram")
    if args.chromagram: print("chromagram")

    json_file_names = os.listdir(label_dir)
    json_file_names = [j for j in json_file_names if j.endswith('.json')]

    wav_file_names = os.listdir(data_dir)
    wav_file_names = [w for w in wav_file_names if w.endswith('.wav')]

    wav_label_dict = dict()
    for json_file_name in json_file_names:
        matched_wav_file = list(filter(lambda x: x[:-3] == json_file_name[:-4], wav_file_names))

        if len(matched_wav_file) == 0: continue
        
        wav_label_dict[json_file_name] = matched_wav_file[0]
    

    if not os.path.isdir(save_region_dir): os.mkdir(save_region_dir)

    for i, (label_file, data_file) in enumerate(wav_label_dict.items(),1):
        if i == 10: break
        print(f"\r{i}/{len(wav_label_dict)}", end="")
        preprocessing(label_dir, data_dir, label_file, data_file, save_region_dir, args)
    print()
    
    


def preprocessing(label_dir, data_dir, label_file, data_file, save_region_dir, args):

    file_name = label_file[:-5]

    TS = 16000*5
    preprocessor = Preprocessor()

    jsons = loadJson(os.path.join(label_dir, label_file))

    durations = getDurations(jsons)
    
    audio, sr = librosa.load(os.path.join(data_dir, data_file), sr=16000)
    print("Audio Load Finish")

    splited_audios = split_audio(audio, sr, durations)
    splited_audios = [audio for audio in splited_audios if len(audio) > 3*sr]
    splited_audios = [preprocessor.resize(audio, TS) for audio in splited_audios]
    padded_audios = np.vstack(splited_audios)

    print(f"Audios shape : {padded_audios.shape}")

    save_path = os.path.join(save_region_dir, file_name)
    if not os.path.isdir(save_path): os.mkdir(save_path)
    if args.chromagram:
        chromas = [preprocessor.chromagram(audio, sr) for audio in padded_audios]
        chromas = np.stack(len(chromas), axis=0)
        preprocessor.saveChromaFig(chromas[0], os.path.join(save_path, f"{file_name}.png"))
        preprocessor.saveChromaAsPickle(chromas, os.path.join(save_path, f"{file_name}.pickle"))
    if args.spectrogram:
        spectros = [preprocessor.spectrogram(audio, sr) for audio in padded_audios]
        spectros = np.stack(len(spectros), axis=0)
        preprocessor.saveSpectroFig(spectros[0], os.path.join(save_path, f"{file_name}.png"))
        preprocessor.saveSpectroAsPickle(spectros, os.path.join(save_path, f"{file_name}.pickle"))        
    if args.mfcc:
        mfcc = [preprocessor.mfcc(audio, sr) for audio in padded_audios]
        mfcc = np.stack(len(mfcc), axis=0)
        preprocessor.saveMFCCFig(mfcc[0], os.path.join(save_path, f"{file_name}.png"))
        preprocessor.saveMFCCAsPickle(mfcc, os.path.join(save_path, f"{file_name}.pickle"))          

if __name__ == '__main__':
    region = 'gangwon'
    main(base_dir = f'../dataset/{region}/', label_dir=f'{region}_label', data_dir=f'{region}_data_1', save_region_dir=f'./{region}_preprocessed')

    

