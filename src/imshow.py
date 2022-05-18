import librosa
from matplotlib import pyplot  as plt
import librosa.display


class PlotAudio:

    def __init__(self, y, sr):
        self.y = y
        self.sr = sr
    
    def waveshow(self):
        librosa.display.waveshow(self.y, self.sr)
        plt.close()

def main():
    data_path = '../dataset/jeju/jeju_data_1/DZES20000002.wav'

    y, sr = librosa.load(data_path, offset=5, duration=3)

    pla = PlotAudio(y, sr)

    pla.waveshow()
    

if __name__ == '__main__':
    main()