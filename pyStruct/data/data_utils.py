""" Some small functions """
from scipy.fft import fft, fftfreq


 
def get_psd(dt_s, x):
    """
    Get fft of the samples
    :param dt_s: time step
    :param x: samples
    :return:
    """

    # 2-side FFT
    N = len(x)
    xdft = fft(x)
    freq = fftfreq(N, dt_s)

    # convert 2-side to 1-side
    if N % 2 == 0:
        xdft_oneside = xdft[0:int(N / 2 )]
        freq_oneside = freq[0:int(N / 2 )]
    else:
        xdft_oneside = xdft[0:int((N - 1) / 2)+1]
        freq_oneside = freq[0:int((N - 1) / 2)+1]


    # Power spectrum
    Fs = 1 / dt_s
    psdx = 1 / (Fs * N) * abs(xdft_oneside)**2
    psdx[1:-1] = 2 * psdx[1:-1] # power for one-side
    return freq_oneside, psdx

