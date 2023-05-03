import emd
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, signal

sample_rate = 1000
seconds = 10
num_samples = sample_rate * seconds

time_vect = np.linspace(0, seconds, num_samples)

freq = 5

# Sinusoidal signal
x = np.cos(2 * np.pi * freq * time_vect)

# Non-linear signal
y = np.cos(2 * np.pi * freq * time_vect) + 0.25 * np.cos(
    2 * np.pi * freq * 2 * time_vect - np.pi
)

z = emd.simulate.ar_oscillator(25, sample_rate, seconds, r=0.975)[:, 0]
imf = emd.sift.sift(z)
IP, IF, IA = emd.spectra.frequency_transform(imf, sample_rate, "hilbert")

freq_range = (0.1, 50, 48)
hht_f, hht = emd.spectra.hilberthuang(
    IF, IA, freq_range, mode="amplitude", sum_time=False
)
print(hht_f.shape, hht.shape, time_vect.shape)

plt.pcolormesh(time_vect, hht_f, hht, cmap="hot_r")
plt.ylim(0, 50)
plt.title("Hilbert-Huang Transform")
plt.xlim(1, 9)
plt.xlabel("Time (seconds)")
plt.ylabel("Frequency (Hz)")
plt.show()
