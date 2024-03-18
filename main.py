from forberedelse import Forberedelse
from optikkLab import OptikkLab, Videodata

def plotTidsakse(datasett: OptikkLab) -> None:
    """Plotter tidsaksen til RGB-verdiene i en av videoene. High- og lowpass
    kan skrus av og på med parametrene, og indeksen i datasett.data[0] bestemmer hvilken 
    video i mappen plottet skal ta fra.

    Args:
        datasett (OptikkLab): Et ferdig initialisert datasett-objekt.
    """
    
    datasett.plotTimeSignal(
        datasett.data[0],
        plotTitle = "RGB tidsserie for scenario 1",
        lowpass = True,
        highpass = True,
        seconds = 7
    )

def FFT(datasett: OptikkLab, farge) -> None:
    """Plotter frekvensspekteret til en av videoene i mappen.
    Fargen frekvensspekteret er basert på kan velges med color = "red", "green", 
    og "blue". Hanningvindu og filtere kan skrus av og på med parametrene.

    Args:
        datasett (OptikkLab): Ferdig initialisert datasett-objekt.
    """
    
    datasett.FFT(
        video = datasett.data[0],
        color = farge,
        highpass = True,
        lowpass = True, 
        hanning = False
    )

def finnPuls(datasett: OptikkLab) -> float:
    """Finner pulsen i videoen gjennom frekvensspekteret. Returnerer en float som 
    representerer pulsen. Fargen pulsen hentes ut i fra kan settes med 
    color ="red", "green", "blue", og videoen kan velges med indeksen til 
    datasett.data[n].

    Args:
        datasett (OptikkLab): Ferdig initialisert datasett-objekt.
    Returns:
        float: Den identifiserte pulsen fra den valgte videoen.
    """

    return datasett.getPulse(
                    datasett.data[1],
                    color = "red"
                    )               
    

if __name__ == "__main__":
    
    videomappe = "Hoy" #Endre til mappen der videoene du skal analysere er
    
    # Velg denne linjen om du skal analysere videoene i en mappe for første gang. Altså med
    # å velge fokusområde på skjermen osv.
    
    datasett = OptikkLab(videomappe) 
    
    # Velg linjen under om du allerede har analysert videoene. Da kan datasettet initialiseres
    # fra en pkl-fil i steden så du slipper det.
    
    #datasett = OptikkLab.fromPikle(videomappe) 
    
    

    i = 1
    for video in datasett.data:
        datasett.plotTimeSignal(video, f"RGB time series", lowpass=True, highpass=True, seconds=7)
        datasett.FFT(video, "red", lowpass=True, highpass=True, plot=True, nr=i) # Plotter frekvensspekter av rød kanal
        datasett.FFT(video, "green", lowpass=True, highpass=True, plot=True, nr=i) # Plotter frekvensspekter av grønn kanal
        datasett.FFT(video, "blue", lowpass=True, highpass=True, plot=True, nr=i) # Plotter frekvensspekter av blå kanal
        print(f"\nPuls fra kanal R i video {i}: {datasett.getPulse(video, "red")} pr. min.")
        print(f"Puls fra kanal G i video {i}: {datasett.getPulse(video, "green")} pr. min.")
        print(f"Puls fra kanal B i video {i}: {datasett.getPulse(video, "blue")} pr. min.\n")
        i += 1
    
    mean, std, SNR = datasett.meanAndStdDevSNR() # Regner ut gjennomsnitt og standardavvik.
    print(f"Gjennomsnitlig puls for alle videoene pr. fargekanal:\n")
    print(f"\tR: {mean[0]}\n\tG: {mean[1]}\n\tB: {mean[2]}\n")
    print(f"Standardavvik for alle videoene pr. fargekanal:\n")
    print(f"\tR: {std[0]}\n\tG: {std[1]}\n\tB: {std[2]}\n\n")
    
    for j in range(1, len(datasett.data) +1):
        print(f"SNR for red channel in video {j}: {SNR[0][j-1]}")
        print(f"SNR for green channel in video {j}: {SNR[1][j-1]}")
        print(f"SNR for blue channel in video {j}: {SNR[2][j-1]}")
        print("")