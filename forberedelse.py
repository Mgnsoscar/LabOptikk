import numpy as np
from typing import List

class Forberedelse:
    
    def __init__(self) -> None:
        
        red_wavelength = 600 # Replace with wavelength in nanometres
        green_wavelength = 520 # Replace with wavelength in nanometres
        blue_wavelength = 460 # Replace with wavelength in nanometres

        self.wavelength = np.array([red_wavelength, green_wavelength, blue_wavelength])
        
        self.bvf = 0.01 # Blood volume fraction, average blood amount in tissue
        self.oxy = 0.8 # Blood oxygenation
        
        self.muabo = np.genfromtxt("resources/muabo.txt", delimiter=",")
        self.muabd = np.genfromtxt("resources/muabd.txt", delimiter=",")
        
        # reduced scattering coefficient ($\mu_s^\prime$ in lab text)
        # the numerical constants are thanks to N. Bashkatov, E. A. Genina and
        # V. V. Tuchin. Optical properties of skin, subcutaneous and muscle
        # tissues: A review. In: J. Innov. Opt. Health Sci., 4(1):9-38, 2011.
        # Units: 1/m
        self.musr = 100 * (17.6*(self.wavelength/500)**-4 + 18.78*(self.wavelength/500)**-0.22)


    def mua_blood_oxy(self, x: List[int]) -> np.ndarray: 
        return np.interp(x, self.muabo[:, 0], self.muabo[:, 1])
    
    def mua_blood_deoxy(self, x: List[int]) -> np.ndarray: 
        return np.interp(x, self.muabd[:, 0], self.muabd[:, 1])

    def mu_as(self, bvf: float, oxy: float) -> np.ndarray:
        """Calculates mu_a based on blood volume fraction and blood oxygenation.

        Args:
            bvf (float): Blood volume fraction.
            oxy (float): Blood oxygenation

        Returns:
            np.ndarray: 
        """
        
        # Absorption coefficient ($\mu_a$ in lab text)
        # Units: 1/m
        mua_other = 25 # Background absorption due to collagen, et cetera
        mua_blood = (self.mua_blood_oxy(self.wavelength)*oxy # Absorption due to
                    + self.mua_blood_deoxy(self.wavelength)*(1-oxy)) # pure blood
        mua = mua_blood*bvf + mua_other

        return mua

    def penetrationDepth(self, bloodVolume: float, oxygenation: float) -> np.ndarray:

        mua = self.mu_as(bloodVolume, oxygenation)
        
        return np.sqrt( 1 / (3*(self.musr+mua)*mua) )*1000

    def transmittance(self, 
                      Width: float, 
                      bloodVolume: float, 
                      oxygenation: float
                      ) -> np.ndarray:
        mua = self.mu_as(bloodVolume, oxygenation)
        C = (np.sqrt(3 * (self.musr+mua)*mua) )
        return np.exp(-C*Width)*100

    def showPreparations(self) -> None:
        
        self.__init__()
    
        print("a)\nPenetrasjonsdybde (mm) RGB")
        print(self.penetrationDepth(0.01, 0.8))
        print("")

        print("b)\nProsent transmittans RGB 1 cm finger")
        print(self.transmittance(0.01, 0.01, 0.8))
        print("")

        print("c)\nProsent transmittans 300 um blodåre 100% blv, RGB")
        hoyBlv = self.transmittance(300e-6, 1, 0.8)
        print(hoyBlv)
        print("")

        print("d)\nProsent transmittans 300 um vev 1% blv, RGB")
        lavBlv = self.transmittance(300e-6, 0.01, 0.8)
        print(lavBlv)
        print("")

        print("Kontrast:")
        kontrast = abs(hoyBlv-lavBlv) / lavBlv
        print(kontrast)

        print("\ne)\nMan burde bruke grønt eller blått lys til å måle puls, siden disse her høyest kontrast.")

        print(
        """\nOppgave 2:

        Man kan definere SNR i pulssammenheng som maksamplituden til 
        pulssignalet delt på standardavviket til støysignalet.

        Om man vet frekvensen på pulsen kan man ta fft av signalet og dele
        gjennomsnitt av frekvensområdet til pulsen og dele på gjennomsnittet 
        av det antatte frekvensområdet til støyen.

        SNR = snitt pulsfrekvenser / snitt av støyfrekvenser
        """
        )
        
if __name__ == "__main__":
    
    Forberedelse().showPreparations()