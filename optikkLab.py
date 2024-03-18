from __future__ import annotations
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt
import cv2
import pickle
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
import statistics
from scipy.signal import detrend

@dataclass
class Videodata:
    name: str
    red: np.ndarray
    green: np.ndarray
    blue: np.ndarray
    samplePeriod: float
    timeAxis: np.ndarray
    
class OptikkLab:
    
    data: List[Videodata]

    def __init__(self, scenario: str) -> None:
        
        self.scenario = scenario
        
        if not os.path.exists(f"resources/results/{scenario}"):
            os.makedirs(f"resources/results/{scenario}")

        if len(os.listdir(f"resources/results/{scenario}")) > 0:
            question = True
            while question:
                answer = input("Do you wish to override previous datafiles for this scenario? y/n?")
                if answer == "y":
                    question = False
                elif answer == "n":
                    raise RuntimeError("Video analysis cancelled.")

        videoFiler = os.listdir(f"videoer/{scenario}")
        for video in videoFiler:

            basename, _ = os.path.splitext(video)
            
                        
            self.analyserVideo(video, basename, scenario)

        dataFiler = os.listdir(f"resources/results/{scenario}")
        
        self.data = []

        for dataFil in dataFiler:

            basename, _ = os.path.splitext(dataFil)

            red     = []
            green   = []
            blue    = []

            if basename == "Pulsresultater":
                continue
            
            with open(f"resources/results/{scenario}/{dataFil}", "r") as file:

                for line in file:

                    line = line.split()

                    if line[0] == "Fps":
                        fps = float(line[1])
                        samplePeriod = 1/fps
                    else:
                        red.append(float(line[0]))
                        green.append(float(line[1]))
                        blue.append(float(line[2]))
            
            currentVideo = Videodata(
                name = basename,
                red = np.array(red),
                green = np.array(green),
                blue = np.array(blue),
                samplePeriod = 1 / fps,
                timeAxis = np.arange(0, len(red) * samplePeriod, samplePeriod)
            )
            
            self.data.append(currentVideo)
        
        with open(f"resources/pikles/{scenario}.pkl", 'wb') as file:
            pickle.dump(self, file)

    def plotTimeSignal(self, 
                       video: Videodata, 
                       plotTitle: str,
                       lowpass: bool = False,
                       highpass: bool = False,
                       seconds: float = 30
                       ) -> None:

        xLabel  =   "Time [s]"
        yLabel  =   "Relative amplitude"

        red = detrend(video.red)
        green = detrend(video.green)
        blue = detrend(video.blue)
        
        if lowpass:
            red = self.butter_lowpass(red, video.samplePeriod)
            green = self.butter_lowpass(green, video.samplePeriod)
            blue = self.butter_lowpass(blue, video.samplePeriod)
        if highpass:
            red = self.butter_highpass(red, video.samplePeriod)
            green = self.butter_highpass(green, video.samplePeriod)
            blue = self.butter_highpass(blue, video.samplePeriod)
        
        yPlots  =   {
            "Red" : red,
            "Green": green,
            "Blue" : blue
        }

        self.plot(video.timeAxis, yPlots, plotTitle, xLabel, yLabel)

        plt.xlim(left = 0, right = seconds)
        plt.savefig(f"resources/results/figures/{self.scenario}/{video.name}_timeSeries")
        #plt.show()

    def FFT(self, 
            video: Videodata, 
            color: str, 
            lowpass: bool = False,
            highpass: bool = False,
            plot: bool = True,
            hanning: bool = False,
            nr: int = 1
            ) -> Optional[Tuple[np.ndarray]]:

        if color not in ["red", "green", "blue"]:
            raise ValueError(f"'{color}' not a valid color. Choose 'red', 'green', or 'blue'.")
        
        signal = detrend(video.__dict__[color])
        
        if hanning:
            signal = signal * np.hamming(len(signal)) 

        if lowpass:
            signal = self.butter_lowpass(signal, video.samplePeriod)
        if highpass:
            signal = self.butter_highpass(signal, video.samplePeriod)

        # Calculate the frequency spectrum using FFT
        freq = np.fft.fftfreq(
            n = len(signal), 
            d = video.samplePeriod
            )
        freq = np.fft.fftshift(freq)

        spectrum = np.fft.fftshift(np.fft.fft(signal))
        spectrum = np.abs(spectrum)
        spectrum = spectrum/np.max(spectrum)
        
        farger = {"red" : "R", "green":"G", "blue":"B"}
        
        if plot:
            self.plot(
                xVal = freq, 
                yVals = {color : 10*np.log10(abs(spectrum)**2)}, 
                title = f"Power spectrum of channel {farger[color]}", 
                xLabel = "Frequency [Hz]",  
                yLabel = "Power [dB]", 
                log = True
                )

            try:
                plt.ylim(bottom=np.min(10*np.log10(abs(spectrum[0:len(spectrum)//2])**2)))
            except:
                plt.ylim(bottom = -80)
            
            plt.savefig(f"resources/results/figures/{self.scenario}/{video.name}_FFT_{color}")
            #plt.show()
        else:
            return freq, spectrum

    def plot(self, xVal, yVals: dict, title, xLabel, 
             yLabel, log=False, figsize=(20,12)) -> None:

        font = {'family': 'serif', 'color': 'darkred', 
                'weight': 'normal', 'size': 16,}

        # Plot the Bode plot
        plt.figure(figsize=figsize)
        plt.title(title, fontsize=35, fontdict=font, y=1.05)
        plt.xlabel(xLabel, fontsize=30, fontdict=font)
        plt.ylabel(yLabel, fontsize=30, fontdict=font)

        for key in yVals:
            if log == True:
                plt.semilogx(xVal, yVals[key], label=key)
            else:
                plt.plot(xVal, yVals[key], label=key, color=key)

        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend(fontsize=20)
    
    def getPulse(self, video: Videodata, color: str) -> float:
                
        frequencies, spectrum = self.FFT(
            video = video,
            color = color,
            lowpass = True,
            highpass = True,
            plot = False,
            hanning = False
            )
        
        maxSpectrumIndex = np.argmax(spectrum)
        maxFrequency = abs(frequencies[maxSpectrumIndex])
        pulse = maxFrequency * 60
        
        return round(pulse, 2)
    
    def meanAndStdDevSNR(self) -> Tuple[List, List, List]:
        
        red, green, blue = [], [], []
        SNRRed, SNRGreen, SNRBlue = [], [], []
        for video in self.data:
            red.append(self.getPulse(video, "red"))
            green.append(self.getPulse(video, "green"))
            blue.append(self.getPulse(video, "blue"))
            SNRRed.append(self.SNR(video.red, video.samplePeriod))
            SNRGreen.append(self.SNR(video.green, video.samplePeriod))
            SNRBlue.append(self.SNR(video.blue, video.samplePeriod))
        
        mean = [np.mean(red), np.mean(green), np.mean(blue)]
        
        try:
            std = [statistics.stdev(red),
            statistics.stdev(green),
            statistics.stdev(blue)]
        except:
            std = [0, 0, 0]

        
        return mean, std, [SNRRed, SNRGreen, SNRBlue]
    
    def SNR(self, spectrum: np.ndarray, sample_period) -> float:

        signal = detrend(spectrum)
        
        signal = signal * np.hamming(len(signal)) 

        signal = self.butter_lowpass(signal, sample_period)

        signal = self.butter_highpass(signal, sample_period)

        # Calculate the frequency spectrum using FFT
        freq = np.fft.fftfreq(
            n = len(signal), 
            d = sample_period
            )
        freq = np.fft.fftshift(freq)

        spectrum = np.fft.fftshift(np.fft.fft(signal))
        spectrum = spectrum/np.max(spectrum)
        spectrum = 10*np.log10(np.abs(spectrum)**2)
        
        
        # Define the frequency range
        freq_range = (30/60, 216/60)  # Hz

        # Identify the indices corresponding to the frequency range of interest
        freq_indices = np.where((abs(freq) >= freq_range[0]) & (abs(freq) <= freq_range[1]), 1, 0)
        
        # Extract the frequencies and corresponding magnitudes within this range
        spectrum_range_values = np.array([i for i, j in zip(spectrum, freq_indices) if j != 0])
        noise_values = np.array([i for i, j in zip(spectrum, freq_indices) if j == 0])
        

        return np.max(spectrum_range_values) - np.mean(noise_values)

    
    @staticmethod
    def butter_highpass(signal: np.ndarray, 
                        samplePeriod: float, 
                        order: int = 5,
                        cutoff: float = 0.5
                        ) -> np.ndarray:

        nyquist = 0.5 * (1/samplePeriod)
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        filtered_data = filtfilt(b, a, signal)
        return filtered_data

    @staticmethod
    def butter_lowpass(signal: np.ndarray, 
                       samplePeriod: float, 
                       order: int = 5,
                       cutoff: float = 3.6
                       ) -> np.ndarray:

        nyquist = 0.5 * (1/samplePeriod)
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        filtered_data = filtfilt(b, a, signal)
        return filtered_data

    @staticmethod
    def analyserVideo(inputVideo: str, outputFil: str, scenario: str) -> None:
            
        filename        = f"videoer/{scenario}/{inputVideo}"
        output_filename = f"resources/results/{scenario}/{outputFil}.txt"

        #read video file
        cap = cv2.VideoCapture(filename, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print("Could not open input file. Wrong filename, or your OpenCV package might not be built with FFMPEG support. See docstring of this Python script.")
            exit()

        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        mean_signal = np.zeros((num_frames, 3))

        #loop through the video
        count = 0
        while cap.isOpened():
            ret, frame = cap.read() #'frame' is a normal numpy array of dimensions [height, width, 3], in order BGR
            if not ret:
                break

            #display window for selection of ROI
            if count == 0:
                window_text = 'Select ROI by dragging the mouse, and press SPACE or ENTER once satisfied.'
                ROI = cv2.selectROI(window_text, frame) #ROI contains: [x, y, w, h] for selected rectangle
                cv2.destroyWindow(window_text)
                print("Looping through video.")

            #calculate mean
            cropped_frame = frame[ROI[1]:ROI[1] + ROI[3], ROI[0]:ROI[0] + ROI[2], :]
            mean_signal[count, :] = np.mean(cropped_frame, axis=(0,1))
            count = count + 1

        cap.release()

        #save to file in order R, G, B.
        np.savetxt(output_filename, np.flip(mean_signal, 1))

        print("Data saved to '" + output_filename + "', fps = " + str(fps) + " frames/second")

        with open(output_filename, 'a') as file:
            file.write(f"Fps {str(fps)}")

    @classmethod
    def fromPikle(cls, piklefile: str) -> OptikkLab:

        with open(f"resources/pikles/{piklefile}.pkl", 'rb') as file:
            instance = pickle.load(file)
        return instance

if __name__ == "__main__":

    ###############################################
    # ANALYSER ET NYTT SETT MED DATA
    ###############################################

    datasett = OptikkLab("Refleksjon_fra_finger")

    ###############################################
    # EVT. HENT INN ANALYSERT DATASETT
    ###############################################

    #datasett = OptikkLab.fromPikle("Gjennom_finger")

    #############################################################
    # PLOT TIDSAKSE. VELG HVILKEN VIDEO I DATASETTET MED INDEKSEN
    #############################################################
    datasett.plotTimeSignal(
        datasett.data[0],
        plotTitle = "RGB time series",
        lowpass = True,
        highpass = True,
        seconds = 7
    )

    #################################################################
    # PLOT FFT. VELG HVILKEN VIDEO I DATASETTET MED INDEKSEN OG VELG 
    # FOR HVILKEN FARGE MED color = "red", "green", eller "blue".
    #################################################################
    datasett.FFT(
        video = datasett.data[0],
        color = "green",
        highpass = True,
        lowpass = True, 
        hanning = False
    )

    #################################################################
    # HENT UT PULSEN. VELG VIDEO MED INDEKSEN OG FARGE MED
    # color = "red", "green", eller "blue".
    #################################################################
    print(datasett.getPulse(
                    datasett.data[1],
                    color = "red"
                    )
        )