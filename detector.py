#! /usr/bin/env python3

from abc import ABC, abstractmethod
import argparse
import dotenv
import logging
import logging.handlers
import numpy as np
from typing import Optional, Tuple
import os
import pathlib
import pyaudio
from scipy.signal import argrelmax
import subprocess
import shlex
import time
import wave


RATE: int = 8000   # サンプリング周波数、フレーム数
CHUNK: int = int(RATE / 2)  # PyAudioで一度に取得するサンプリング数. サンプリング周波数の半分。0.5秒分
FORMAT = pyaudio.paInt16  # フォーマット
CHANNELS: int = 1  # チャンネル数 （モノラル）

FREQ_1ST: float = 849.0  # ピンポーンのピンの周波数
FREQ_2ND: float = 680.0  # ピンポーンのポーンの周波数

WINDOW_IN_SECONDS: float = 1.5  # ピンポーンを検出を試みる区間の秒数(window)
WINDOW_IN_FRAMES: int = int(WINDOW_IN_SECONDS * RATE)  # ピンポーン検出を試みる区間のフレーム数

logger: Optional[logging.Logger] = None


# 音入力のインターフェース
class FrameReader(ABC):
    @abstractmethod
    def open(self) -> bool:
        pass

    @abstractmethod
    def shouldOpenAgain(self) -> bool:
        pass

    @abstractmethod
    def read(self, chunk=CHUNK) -> Optional[np.ndarray]:
        pass

    @abstractmethod
    def close(self) -> bool:
        pass


# 音入力としてwavファイルを利用するときのFrameReader。デバッグ用
class WavFrameReader(FrameReader):
    def __init__(self, wavFilePath: pathlib.Path) -> None:
        super().__init__()

        self._wavFilePath = wavFilePath
        self._wavFile = None

    def open(self) -> bool:
        self._wavFile = wave.open(str(self._wavFilePath), 'rb')
        if not self._wavFile:
            logger.error(f"Failed to open the wav file. {self._wavFilePath}")
            return False

        if self._wavFile.getframerate() != RATE:
            logger.error(f"Invalid framerate . {self._wavFile.getframerate()}")
            return False

        if self._wavFile.getnchannels() != CHANNELS:
            logger.error(f"Invalid channels . {self._wavFile.getnchannels()}")
            return False

        if self._wavFile.getsampwidth() != 2:
            logger.error(f"Invalid width . {self._wavFile.getsamwidth()}")
            return False

        return True

    def shouldOpenAgain(self) -> bool:
        return False

    def read(self, chunk=CHUNK) -> Optional[np.ndarray]:
        frames = self._wavFile.readframes(chunk)
        if not frames:
            return None

        return np.frombuffer(frames, dtype='int16')

    def close(self) -> bool:
        self._wavFile.close()
        self._wavFile = None
        return True


# 音入力としてmicを利用するときのFrameReader。
class MicFrameReader(FrameReader):
    def __init__(self) -> None:
        super().__init__()
        self._p = None
        self._stream = None

    def open(self) -> bool:
        self._p = pyaudio.PyAudio()

        input_device_index = -1

        for host_index in range(0, self._p.get_host_api_count()):
            logger.info(f"host: {self._p.get_host_api_info_by_index(host_index)}")
            for device_index in range(0, self._p.get_host_api_info_by_index(host_index)['deviceCount']):
                device_info = self._p.get_device_info_by_host_api_device_index(host_index, device_index)
                logger.info(f"device: {device_info}")

                if device_info['name'] == os.environ.get("AUDIO_DEVICE"):
                    input_device_index = device_info["index"]
                    break
            else:
                continue
            break

        if input_device_index < 0:
            self.close()
            return False

        logger.info(f"========= {input_device_index}")
        try:
            self._stream = self._p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=input_device_index,
                frames_per_buffer=CHUNK
            )
        except Exception as e:
            logger.error(e)
            self.close()
            return False

        return True

    def shouldOpenAgain(self) -> bool:
        return True

    def read(self, chunk=CHUNK) -> Optional[np.ndarray]:
        if not self._stream.is_active():
            return None

        frames = self._stream.read(CHUNK)
        return np.frombuffer(frames, dtype='int16')

    def close(self) -> bool:
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None

        if self._p:
            self._p.terminate()
            self._p = None

        return True


def setup_logger(name, console=False, level=logging.INFO, logfile='LOGFILENAME.txt') -> logging.Logger:

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    fmt = "%(asctime)s %(thread)d %(levelname)s %(name)s :%(message)s"

    # create file handler which logs even DEBUG messages
    fh = logging.handlers.RotatingFileHandler(logfile, maxBytes=1000000, backupCount=10)
    fh.setLevel(level)
    fh_formatter = logging.Formatter(fmt)
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    if console:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch_formatter = logging.Formatter(fmt)
        ch.setFormatter(ch_formatter)
        logger.addHandler(ch)

    return logger


# FFTをする。
def fft(frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    x = np.fft.fft(frames)
    freq = np.fft.fftfreq(len(frames), d=1.0 / RATE)

    x = x[:int(len(x) / 2)]
    freq = freq[:int(len(freq) / 2)]
    amp = np.sqrt(x.real ** 2 + x.imag ** 2)

    return (amp, freq)


# Peakを求める。
def findpeaks(x: np.ndarray, y: np.ndarray, n: int = 50, w: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    index_all = argrelmax(y, order=w)                # scipyのピーク検出
    index = []                                                      # ピーク指標の空リスト
    peaks = []                                                      # ピーク値の空リスト

    # n個分のピーク情報(指標、値）を格納
    for i in range(min(len(index_all[0]), n)):
        index.append(index_all[0][i])
        peaks.append(y[index_all[0][i]])
    index = np.array(index) * x[1]                                  # xの分解能x[1]をかけて指標を物理軸に変換
    peaks = np.array(peaks)

    return index, peaks


# ピークに、指定された周波数が含まれるかチェック
def has_freq(freq: np.ndarray, target: float) -> bool:
    l = freq.tolist()
    res = list(filter(lambda x: target - 5 <= x and x <= target + 5, l))
    return bool(res)


# Alexa Echo に 話させる。
def speak_alexa() -> None:
    device = os.environ.get("ALEXA_DEVICE")
    cmd = f'./alexa_remote_control.sh -d "{device}" -e "speak: チャイムが鳴りました"'
    logging.debug(cmd)
    # cmd = './alexa_remote_control.sh -d "ALL" -e "speak: チャイムが鳴りました"'
    subprocess.run(shlex.split(cmd))


def main() -> None:
    dotenv.load_dotenv(verbose=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", help="verbose", action="count", default=0)
    parser.add_argument("-c", "--console", help="console output", action='store_true')
    args = parser.parse_args()

    logging_level = logging.INFO if args.verbose == 0 else logging.DEBUG
    global logger
    logger = setup_logger(__name__, args.console, logging_level, "intercom.log")

    # frame_reader: FrameReader = WavFrameReader('test-data/sample1.wav')
    frame_reader: FrameReader = MicFrameReader()

    # Open FrameReader
    while True:
        logger.info("Try to open")
        if frame_reader.open():
            logger.info("Success")
            break

        if not frame_reader.shouldOpenAgain():
            frame_reader.close()
            return

        time.sleep(1)

    # Read frames
    counter: int = 0
    skip_count: int = 0
    window_frames = None
    while True:
        frames = frame_reader.read()
        logger.debug(f"frames: {frames}")
        if counter == 0:
            logger.info(".")
        counter = counter + 1 if counter < 10 else 0

        if frames is None:
            logger.info("End of frames")
            break

        skip_count = max(0, skip_count - 1)
        if window_frames is None:
            window_frames = frames
        else:
            window_frames = np.concatenate([window_frames, frames])

        window_frames = window_frames[max(0, len(window_frames) - WINDOW_IN_FRAMES):]

        if len(window_frames) == WINDOW_IN_FRAMES:
            # FFT
            amp, freq = fft(window_frames)

            # Find peaks
            index, peaks = findpeaks(freq, amp)

            # Detect
            if has_freq(index, FREQ_1ST) and has_freq(index, FREQ_2ND):
                logger.info(f"detect!!! {skip_count}")

                if skip_count == 0:
                    speak_alexa()

                skip_count = 5

    frame_reader.close()


if __name__ == "__main__":
    main()

    # dotenv.load_dotenv(verbose=True)
    # speak_alexa()
