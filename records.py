#! /usr/bin/env python3

import wave
import pyaudio

RATE: int = 8000
CHUNK: int = int(RATE / 2)
FORMAT = pyaudio.paInt16
CHANNELS: int = 1
Record_Seconds: int = 10


def main() -> None:
    p = pyaudio.PyAudio()

    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    print("now recording...")
    all = []

    for i in range(0, int(RATE / CHUNK * Record_Seconds)):
        frames = stream.read(CHUNK)
        print(f"-------------- {type(frames)}, {len(frames)}, {frames[0]}, {frames[1]}")
        all.append(frames)

    print("Finished Recording...")

    stream.close()
    p.terminate()

    wavFile = wave.open("output.wav", "wb")
    wavFile.setnchannels(CHANNELS)
    wavFile.setsampwidth(p.get_sample_size(FORMAT))
    wavFile.setframerate(RATE)
    wavFile.writeframes(b"".join(all))
    wavFile.close()


if __name__ == "__main__":
    main()
