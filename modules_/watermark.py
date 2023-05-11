import numpy as np
from scipy.io import wavfile
from scipy.signal import windows
import pywt

AUDIO_INPUT = "audio.wav"  # Watermark destination file
AUDIO_WATERMARKED = "wmed_signal.wav"  # Files with embedded watermarks
PSEUDO_RAND_FILE = "pseudo_rand.dat"  # file of pseudo-random number sequences
WATERMARK_ORIGINAL_FILE = "watermark_ori.dat"  # watermark signal

REP_CODE = True  # # Use repeating padding
FRAME_LENGTH = 2048  # frame length
CONTROL_STRENGTH = 1000  # Embedding Strength
OVERLAP = 0.5  # Frame Analysis Overlap Rate (Fixed)
NUM_REPS = 3  # number of repetitions of embedding

WAVELET_BASIS = "db4"
WAVELET_LEVEL = 3
WAVELET_MODE = "symmetric"
THRESHOLD = 0.0


def fix(xs):
    """
    A emuration of MATLAB 'fix' function.
    borrowed from https://ideone.com/YjJwOh
    """

    # res = [np.floor(e) if e >= 0 else np.ceil(e) for e in xs]
    if xs >= 0:
        res = np.floor(xs)
    else:
        res = np.ceil(xs)
    return res


def embed():
    """
    perform embedding.
    """

    # Host Signals
    sr, host_signal = wavfile.read(AUDIO_INPUT)
    signal_len = len(host_signal)

    # Frame movement
    frame_shift = int(FRAME_LENGTH * (1 - OVERLAP))

    # Overlap length with adjacent frames
    overlap_length = int(FRAME_LENGTH * OVERLAP)

    # Total number of padding bits
    embed_nbit = fix((signal_len - overlap_length) / frame_shift)

    if REP_CODE:
        # Effective number of pedgable bits
        effective_nbit = np.floor(embed_nbit / NUM_REPS)

        embed_nbit = effective_nbit * NUM_REPS
    else:
        effective_nbit = embed_nbit

    # Integerization
    frame_shift = int(frame_shift)
    effective_nbit = int(effective_nbit)
    embed_nbit = int(embed_nbit)

    # Create original watermark signals (0 and 1 bit strings)
    wmark_original = np.random.randint(2, size=int(effective_nbit))

    # Save original watermark signals
    with open(WATERMARK_ORIGINAL_FILE, "w") as f:
        for d in wmark_original:
            f.write("%d\n" % d)

    # Extend watermark signal
    if REP_CODE:
        wmark_extended = np.repeat(wmark_original, NUM_REPS)
    else:
        wmark_extended = wmark_original

    # Watermark embedding strength
    alpha = CONTROL_STRENGTH

    # Generate a watermarked signal in wavelet domain
    pointer = 0
    count = 0
    wmed_signal = np.zeros((frame_shift * embed_nbit))  # watermarked signal
    prev = np.zeros((FRAME_LENGTH))
    for i in range(embed_nbit):
        frame = host_signal[pointer : pointer + FRAME_LENGTH]

        # Calculate Wavelet Coefficient
        coeffs = pywt.wavedec(
            data=frame, wavelet=WAVELET_BASIS, level=WAVELET_LEVEL, mode=WAVELET_MODE
        )

        # Set the watermark embedding strength to the same order as the average (adaptive)
        # coef_size = int(np.log10(np.abs(np.mean(coeffs[0])))) + 1
        # alpha = 10 ** coef_size

        # Embedding watermarks
        if wmark_extended[count] == 1:
            coeffs[0] = coeffs[0] - np.mean(coeffs[0]) + alpha
        else:
            coeffs[0] = coeffs[0] - np.mean(coeffs[0]) - alpha

        # Reconfiguration
        wmarked_frame = pywt.waverec(
            coeffs=coeffs, wavelet=WAVELET_BASIS, mode=WAVELET_MODE
        )

        # # Hann window
        wmarked_frame = wmarked_frame * windows.hann(FRAME_LENGTH)

        wmed_signal[frame_shift * i : frame_shift * (i + 1)] = np.concatenate(
            (
                prev[frame_shift:FRAME_LENGTH] + wmarked_frame[0:overlap_length],
                wmarked_frame[overlap_length:frame_shift],
            )
        )

        prev = wmarked_frame
        count = count + 1
        pointer = pointer + frame_shift

    # Merge with the rest of the host signal
    wmed_signal = np.concatenate(
        (wmed_signal, host_signal[len(wmed_signal) : signal_len])
    )

    # Save signal with embedded watermark as WAV
    wmed_signal = wmed_signal.astype(np.int16)  # convert float into integer
    wavfile.write(AUDIO_WATERMARKED, sr, wmed_signal)


def detect():
    """
    perform detecton.
    """

    sr, host_signal = wavfile.read(AUDIO_INPUT)

    # Open an embedded audio file
    _, eval_signal = wavfile.read(AUDIO_WATERMARKED)
    signal_len = len(eval_signal)

    # Load original watermark signal
    with open(WATERMARK_ORIGINAL_FILE, "r") as f:
        wmark_original = f.readlines()
    wmark_original = np.array([float(w.rstrip()) for w in wmark_original])

    # Frame movement
    frame_shift = int(FRAME_LENGTH * (1 - OVERLAP))

    # Number of padding bits
    embed_nbit = fix((signal_len - FRAME_LENGTH * OVERLAP) / frame_shift)

    if REP_CODE:
        # Actual number of embedding bits
        effective_nbit = np.floor(embed_nbit / NUM_REPS)

        embed_nbit = effective_nbit * NUM_REPS
    else:
        effective_nbit = embed_nbit

    frame_shift = int(frame_shift)
    effective_nbit = int(effective_nbit)
    embed_nbit = int(embed_nbit)

    # Load original watermark signal
    with open(WATERMARK_ORIGINAL_FILE, "r") as f:
        wmark_original = f.readlines()
    wmark_original = np.array([int(w.rstrip()) for w in wmark_original])

    # Detecting watermark information
    pointer = 0
    detected_bit = np.zeros(embed_nbit)
    for i in range(embed_nbit):
        wmarked_frame = eval_signal[pointer : pointer + FRAME_LENGTH]

        # wavelet decomposition
        wmarked_coeffs = pywt.wavedec(
            data=wmarked_frame,
            wavelet=WAVELET_BASIS,
            level=WAVELET_LEVEL,
            mode=WAVELET_MODE,
        )

        thres = np.sum(wmarked_coeffs[0])

        if thres >= THRESHOLD:
            detected_bit[i] = 1
        else:
            detected_bit[i] = 0

        pointer = pointer + frame_shift

    if REP_CODE:
        count = 0
        wmark_recovered = np.zeros(effective_nbit)

        for i in range(effective_nbit):
            # Aggregate bits (average)
            ave = np.sum(detected_bit[count : count + NUM_REPS]) / NUM_REPS

            if ave >= 0.5:  # majority
                wmark_recovered[i] = 1
            else:
                wmark_recovered[i] = 0

            count = count + NUM_REPS
    else:
        wmark_recovered = detected_bit

    # Display bit error rate
    denom = np.int(np.sum(np.abs(wmark_recovered - wmark_original)))
    BER = np.sum(np.abs(wmark_recovered - wmark_original)) / effective_nbit * 100
    print(f"BER = {BER} % ({denom} / {effective_nbit})")

    # Show SNR
    SNR = 10 * np.log10(
        np.sum(np.square(host_signal.astype(np.float32)))
        / np.sum(
            np.square(host_signal.astype(np.float32) - eval_signal.astype(np.float32))
        )
    )
    print(f"SNR = {SNR:.2f} dB")

    # Show bps
    print("BPS = {:.2f} bps".format(embed_nbit / (len(host_signal) / sr)))


def main():
    """Main routine."""
    embed()
    detect()


if __name__ in "__main__":
    main()
