from multiprocessing import Process, Event, Queue
from queue import Empty
from argparse import ArgumentParser
from pathlib import Path
import sys
from datetime import datetime
import time
import numpy as np
import cv2 as cv
from tqdm import tqdm
from scipy.stats import circvar, circstd
import matplotlib.pyplot as plt
import pyaudio
import wave
import pickle
from contextlib import contextmanager
from ctypes import CFUNCTYPE, c_char_p, c_int, cdll
from utils import MvTracker, get_video_meta, VideoIterator
import sys
import librosa
import os
import multiprocessing
from typing import Union
import subprocess as sub

ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)

def py_error_handler(filename, line, function, err, fmt):
    pass

c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)

@contextmanager
def noalsaerr():
    asound = cdll.LoadLibrary('libasound.so')
    asound.snd_lib_error_set_handler(c_error_handler)
    yield
    asound.snd_lib_error_set_handler(None)

def get_available_cameras():
    available_cameras = []
    # Check for 5 cameras 
    for i in range(5):
        cap = cv.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

def get_available_audio_devices():
    with noalsaerr():
        p = pyaudio.PyAudio()
        devices = []
        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            # rate = p.get_device_info_by_index(i).get('defaultSampleRate')
            print(f"index:[{i}]: {dev.get('name')}")
            print(dev)
            if dev['maxInputChannels'] > 0:
                devices.append(dev['name'])
        p.terminate()
    return devices


END_EVT = Event()
START_EVT = Event()
TALK_EVT = Event()
INITVBR_EVT = Event()
STOP_EVT = Event()
AUDIO_Q1 = Queue()
AUDIO_Q2 = Queue()
DISP_Q1 = Queue()
DISP_Q2 = Queue()

def curvature(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(ddx * dy - dx * ddy) / (dx ** 2 + dy ** 2) ** 1.5
    return curvature

class Stream():
    """
    extends [cv2::VideoCapture class](https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html)
    for video or stream subsampling.
    """

    def __init__(self, filename: Union[str, int], target_fps: int = None):
        self.stream_id = filename
        self._cap = cv.VideoCapture(self.stream_id)
        self._cap.set(cv.CAP_PROP_BUFFERSIZE, 2)
        self.frameWidth = 1920 
        self.frameHeight = 1080
        if not self.isOpened():
            raise FileNotFoundError("Stream not found")

        self.target_fps = target_fps
        self.fps = None
        self.extract_freq = None
        self.compute_extract_frequency()
        self._frame_index = 0

    def compute_extract_frequency(self):
        """evaluate the frame rate over a period of 5 seconds"""
        self.fps = self._cap.get(cv.CAP_PROP_FPS)
        self.frameWidth = self._cap.get(cv.CAP_PROP_FRAME_WIDTH)
        self.frameHeight = self._cap.get(cv.CAP_PROP_FRAME_HEIGHT)
        #print(f"frameWidth: {self.frameWidth}, frameHeight: {self.frameHeight}, frameFPS: {self.fps}")
        if self.fps == 0:
            self.compute_origin_fps()

        if self.target_fps is None:
            self.extract_freq = 1
        else:
            self.extract_freq = int(self.fps / self.target_fps)

            if self.extract_freq == 0:
                raise ValueError("desired_fps is higher than half the stream frame rate")

    def compute_origin_fps(self, evaluation_period: int = 5):
        """evaluate the frame rate over a period of 5 seconds"""
        while self.isOpened():
            ret, _ = self._cap.read()
            if ret is True:
                if self._frame_index == 0:
                    start = time.time()

                self._frame_index += 1

                if time.time() - start > evaluation_period:
                    break

        self.fps = round(self._frame_index / (time.time() - start), 2)

    def read(self):
        """Grabs, decodes and returns the next subsampled video frame."""
        ret, frame = self._cap.read()
        if ret is True:
            self._frame_index += 1

            if self._frame_index == self.extract_freq:
                self._frame_index = 0
                return ret, frame

        return False, False

    def isOpened(self):
        """Returns true if video capturing has been initialized already."""
        return self._cap.isOpened()

    def release(self):
        """Closes video file or capturing device."""
        self._cap.release()

def capture(dishNum, cameras, duration, debug=False):
    if debug:
        print(f"parent process: {os.getppid()}")
        print(f"process id: {os.getpid()}")

    vc = Stream(cameras[dishNum-1], 5) #cv.VideoCapture(cameras[dishNum-1])
    if not vc.isOpened():
        sys.exit(f"Error opening video stream, dish{dishNum}")

    frameWidth = int(vc.frameWidth) #int(vc.get(cv.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(vc.frameHeight) #int(vc.get(cv.CAP_PROP_FRAME_HEIGHT))
    frameFPS = int(vc.fps)#int(vc.get(cv.CAP_PROP_FPS))
    if debug:
        print(f"frameWidth: {frameWidth}, frameHeight: {frameHeight}, frameFPS: {frameFPS}")

    try:
        while vc.isOpened() and not END_EVT.is_set():
            try:
                ret, frame = vc.read()
                #if not ret:
                #    print("No frame received. Exiting...")
                #    break
                if ret:
                    if dishNum == 1:
                        DISP_Q1.put(frame)
                    else:
                        DISP_Q2.put(frame)
            except KeyboardInterrupt:
                break
    except KeyboardInterrupt:
        pass
    finally:
        print("Closing streams...") 
        vc.release()
        if not END_EVT.is_set():
            END_EVT.set()

def track_display_record(EXP, dishNum, frameWidth, frameHeight, frameFPS, duration, offset_x, offset_y, max_dist, canny_thres, nms_thres, avg_windowSize, HEADING_WINDOW, debug=False):
    if debug:
        print(f"parent process: {os.getppid()}")
        print(f"process id: {os.getpid()}")

    
    outputVideoFile = f'EXP/output_exp{EXP}_dish{dishNum}_f{FREQUENCY}_a{AMPLITUDE}.mp4'
    outputVideoFileOrg = f'EXP/output_org_exp{EXP}_dish{dishNum}_f{FREQUENCY}_a{AMPLITUDE}.mp4'

    # ploting the amplitude of sent vibration
    #plt.ion()
    #fig = plt.figure()
    #ax = fig.add_subplot(1, 1, 1)
    #plt.ylim((0, 1.5))

    tracker = MvTracker(frameWidth, frameHeight, offset_x=offset_x, offset_y=offset_y, max_dist=max_dist, canny_thres=canny_thres, nms_thres=nms_thres, debug=debug)

    cv.namedWindow(f'Camera{dishNum}', cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_NORMAL)

    saveVideo = cv.VideoWriter(outputVideoFile, cv.VideoWriter_fourcc(*'mp4v'), frameFPS, (frameWidth, frameHeight))
    saveVideoOriginal = cv.VideoWriter(outputVideoFileOrg, cv.VideoWriter_fourcc(*'mp4v'), frameFPS, (frameWidth, frameHeight))
    coordinates: dict[int, list] = {}
    velocities: dict[int, list] = {}
    coordinates_times: dict[int, list] = {}
    headings: dict[int, list] = {}
    amplitudes = []
    amplitudes_times = []
    headings_times = []
    times = []
    frames = []
    color = (0, 0, 255)
    startTime = datetime.now()
    msg = False
    volume = 0.0
    frameCount = 0
    window = avg_windowSize
    heading_window = HEADING_WINDOW
    counter = 0
    avg_velocity = 0
    notSet = True
    try:
        while not END_EVT.wait(duration):
            try:
                try:
                    if dishNum == 1:
                        frame = DISP_Q1.get(block=False)
                    else:
                        frame = DISP_Q2.get(block=False)
                except Empty:
                    continue

                frameCount += 1
                
                if frameCount == 1:
                    print(f"Camera{dishNum}: {startTime}")
                    cv.imwrite(f'EXP/output_exp{EXP}_dish{dishNum}_f{FREQUENCY}_a{AMPLITUDE}.jpg', frame)

                if debug:
                    print(f"Dish {dishNum}: {(datetime.now() - startTime).total_seconds()}")

                saveVideoOriginal.write(frame)

                # Track objects
                tracker.track(frame)
                #print(tracker.abs_pos)
                if debug:
                    print(f"Number of detected objects: {len(tracker.tracked_objects)}")
                for obj in tracker.tracked_objects:
                    center = obj.estimate.squeeze(axis=0).astype(int).tolist()
                    #if dishNum == 1:
                    #    print(center)

                    if START_EVT.is_set() and variable_windowSize and notSet:
                        traj_length = np.sum(np.sqrt(np.sum(np.square(np.diff(np.array(coordinates[obj.id]), axis=0)), axis=1)), axis=0)
                        avg_velocity = traj_length / frameCount #pixels/frameRate
                        print(f"dish{dishNum}: {avg_velocity}, {traj_length}, {frameCount}")
                        heading_window = int(HEADING_WINDOW / np.squeeze(avg_velocity))
                        window = int(avg_windowSize / np.squeeze(avg_velocity))
                        notSet = False
                        #window = int(avg_windowSize + 20 * np.clip((1-np.mean(np.sum(np.square(velocities[obj.id][VELOCITY_WINDOW:]), axis=1))), 0, 1))

                    #if center[0] > 1300:
                    #    continue

                    if obj.id not in coordinates:
                        coordinates[obj.id] = [center]
                        coordinates_times[obj.id] = [[datetime.now(), frameCount, center]]
                    else:
                        coordinates[obj.id].append(center)
                        coordinates_times[obj.id].append([datetime.now(), frameCount, center])
                    
                    if len(coordinates[obj.id]) > 3:
                        print(f"dish{dishNum}: {curvature(coordinates[obj.id][-3:][0], coordinates[obj.id][-3:][1])}")
                        
                    if obj.id not in velocities:
                        velocities[obj.id] = [tracker.abs_vels[obj.id][-1]]
                    else:
                        velocities[obj.id].append(tracker.abs_vels[obj.id][-1])

                    if len(coordinates[obj.id]) > heading_window:
                        # Compute the heading
                        #if counter == 0:
                        heading = np.arctan2((coordinates[obj.id][-1][1] - coordinates[obj.id][-heading_window][1]), (coordinates[obj.id][-1][0] - coordinates[obj.id][-heading_window][0]))
                        #    counter = counter + 1
                        #elif counter < heading_window:
                        #    heading = headings[obj.id][-1]
                        #    counter = counter + 1
                        #else:
                        #    counter = 0
                        if obj.id not in headings:
                            headings[obj.id] = [heading]
                        else:
                            headings[obj.id].append(heading)
                            headings_times.append([datetime.now(), frameCount, heading])

                        if len(headings[obj.id]) > window:
                            # compute circular variance of the headings (between 0-2pi)
                            variance = circvar(headings[obj.id][-window:], low=-np.pi, high=np.pi) #circvar
                            #print(f"dish{dishNum}: {variance}")
                            amplitudes.append(1-variance)
                            amplitudes_times.append([datetime.now(), frameCount, 1-variance])
                            #if dishNum == 1:
                            #    AMP_Q2.put(amplitudes[-1])
                            #else:
                            #    AMP_Q1.put(amplitudes[-1])
                            if amplitudes[-1] < THRESHOLD:
                                volume = 0.0
                            else:
                                volume = amplitudes[-1]
 
                            if TALK_EVT.is_set():
                                if dishNum == 1:
                                    AUDIO_Q1.put(volume)
                                else:
                                    AUDIO_Q2.put(volume)
                        else:
                            amplitudes.append(None)
                            amplitudes_times.append([datetime.now(), frameCount, None])
                    else:
                        amplitudes.append(None)
                        amplitudes_times.append([datetime.now(), frameCount, None])

                    for idx in range(1, len(coordinates[obj.id])):
                        cv.line(frame, np.intp(coordinates[obj.id][idx - 1]), np.intp(coordinates[obj.id][idx]), color, 2) #colors[tid%len(colors)][::-1], 2)

                times.append((datetime.now() - startTime).total_seconds())
                frames.append(frameCount)
                cv.putText(frame, "{:.0f}".format((datetime.now() - startTime).total_seconds()), \
                        org=(30, 50), fontFace=cv.FONT_HERSHEY_PLAIN , fontScale=2, \
                        color=(255, 255, 255), thickness=2, lineType=cv.LINE_AA)
                if (datetime.now() - startTime).total_seconds() < (START) * 60:
                    cv.putText(frame, "Vibration OFF", \
                                org=(frameWidth-300, 50), fontFace=cv.FONT_HERSHEY_PLAIN , fontScale=2, \
                                color=(255, 0, 0), thickness=2, lineType=cv.LINE_AA)                
                elif (datetime.now() - startTime).total_seconds() < (START+DURATION) * 60:
                    if dishNum == 1:
                        cv.putText(frame, "Vibration ON", \
                                org=(frameWidth-300, 50), fontFace=cv.FONT_HERSHEY_PLAIN , fontScale=2, \
                                color=(255, 0, 0), thickness=2, lineType=cv.LINE_AA)
                    else:
                        cv.putText(frame, "Vibration OFF", \
                                org=(frameWidth-300, 50), fontFace=cv.FONT_HERSHEY_PLAIN , fontScale=2, \
                                color=(255, 0, 0), thickness=2, lineType=cv.LINE_AA)
                elif (datetime.now() - startTime).total_seconds() < (START+DURATION+TALK) * 60:
                    msgSent = "{:.2f}".format(volume)
                    cv.putText(frame, f"Talking: {msgSent}", \
                    org=(frameWidth-350, 50), fontFace=cv.FONT_HERSHEY_PLAIN , fontScale=2, \
                    color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)
                    text = "{:.2f}".format(headings[obj.id][-1])
                    cv.putText(frame, f"Heading: {text}", \
                    org=(frameWidth-350, 100), fontFace=cv.FONT_HERSHEY_PLAIN , fontScale=2, \
                    color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)
                    text = "{:.0f}".format(window)
                    cv.putText(frame, f"Window: {text}", \
                    org=(frameWidth-350, 150), fontFace=cv.FONT_HERSHEY_PLAIN , fontScale=2, \
                    color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)
                    text = "{:.2f}".format(avg_velocity)
                    cv.putText(frame, f"Velocity: {text}", \
                    org=(frameWidth-350, 200), fontFace=cv.FONT_HERSHEY_PLAIN , fontScale=2, \
                    color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)
                    text = "{:.0f}".format(heading_window)
                    cv.putText(frame, f"Heading Window: {text}", \
                    org=(frameWidth-350, 250), fontFace=cv.FONT_HERSHEY_PLAIN , fontScale=2, \
                    color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)
                else:
                    cv.putText(frame, "Stop Talking", \
                            org=(frameWidth-300, 50), fontFace=cv.FONT_HERSHEY_PLAIN , fontScale=2, \
                            color=(255, 0, 0), thickness=2, lineType=cv.LINE_AA)     

                saveVideo.write(frame)
                frame = cv.resize(frame, None, fx=1, fy=1, interpolation=cv.INTER_NEAREST)
                cv.imshow(f'Camera{dishNum}', frame)
                #plt.plot(amplitudes, color='royalblue', linewidth=2)
                #plt.draw()
                #plt.ylim((0, 1.0))
                #plt.pause(plot_pause)
                #plt.clf()    
                cv.pollKey()
            except KeyboardInterrupt:
                break
    except KeyboardInterrupt:
        pass
    finally:
        print("Closing streams...") 
        saveVideo.release()
        saveVideoOriginal.release()
        cv.destroyAllWindows()
        f = open(f'EXP/coordinates_exp{EXP}_dish{dishNum}_f{FREQUENCY}_a{AMPLITUDE}.pickle', 'wb')
        pickle.dump(coordinates, f)
        f.close()
        f = open(f'EXP/headings_exp{EXP}_dish{dishNum}_f{FREQUENCY}_a{AMPLITUDE}.pickle', 'wb')
        pickle.dump(headings, f)
        f.close()
        f = open(f'EXP/amplitudes_exp{EXP}_dish{dishNum}_f{FREQUENCY}_a{AMPLITUDE}.pickle', 'wb')
        pickle.dump(amplitudes, f)
        f.close()
        f = open(f'EXP/frames_exp{EXP}_dish{dishNum}_f{FREQUENCY}_a{AMPLITUDE}.pickle', 'wb')
        pickle.dump(frames, f)
        f.close()
        f = open(f'EXP/times_exp{EXP}_dish{dishNum}_f{FREQUENCY}_a{AMPLITUDE}.pickle', 'wb')
        pickle.dump(times, f)
        f.close()
        f = open(f'EXP/coordinates_times_exp{EXP}_dish{dishNum}_f{FREQUENCY}_a{AMPLITUDE}.pickle', 'wb')
        pickle.dump(coordinates_times, f)
        f.close()
        f = open(f'EXP/amplitudes_times_exp{EXP}_dish{dishNum}_f{FREQUENCY}_a{AMPLITUDE}.pickle', 'wb')
        pickle.dump(amplitudes_times, f)
        f.close()
        f = open(f'EXP/headings_times_exp{EXP}_dish{dishNum}_f{FREQUENCY}_a{AMPLITUDE}.pickle', 'wb')
        pickle.dump(headings_times, f)
        f.close()
        #fig, ax = plt.subplots()
        #for tid, traj in coordinates.items():
        #    traj = np.array(traj)
        #    ax.plot(traj[:, 0], traj[:, 1], label=f"Object {tid}")
        #ax.set_title(f"Trajectories of objects in {dishNum}")
        #ax.set_xlabel("X coordinate")
        #ax.set_ylabel("Y coordinate")
        #ax.legend()
        #fig.savefig(f"EXP/output_exp{EXP}_dish{dishNum}_f{FREQUENCY}_a{AMPLITUDE}.png")
        #plt.show()
        if not END_EVT.is_set():
            END_EVT.set()


def audio(EXP, dishNum, SAMPLE_RATE, AMPLITUDE, FREQUENCY, frameFPS, speakers, duration, avg_windowSize, debug=False):
    if debug:
        print(f"parent process: {os.getppid()}")
        print(f"process id: {os.getpid()}")

    # apply vibration to the other dish
    outputAudioFile = f'EXP/msg_exp{EXP}_dish{dishNum}_f{FREQUENCY}_a{AMPLITUDE}.wav'
    with noalsaerr():
        p = pyaudio.PyAudio()
    if debug:
        print(f"Dish: {dishNum}, {p.get_device_info_by_index(speakers[2-dishNum])}")
    stream = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=SAMPLE_RATE,
                        output=True, output_device_index=speakers[2-dishNum]) 
    wavefile = wave.open(outputAudioFile, 'wb')
    wavefile.setnchannels(1)
    wavefile.setsampwidth(p.get_sample_size(pyaudio.paFloat32))
    wavefile.setframerate(SAMPLE_RATE)
    startTime = datetime.now()
    #if dishNum == 1:
    #    print(f"waiting to start: {startTime}")
    try:
        START_EVT.wait()
        if dishNum == 1:
            #print(f"waiting for dish1: {datetime.now() - startTime}")
            INITVBR_EVT.wait()
            #print(f"waiting done: {datetime.now() - startTime}")
            #time.sleep((START+DURATION)*60)
        else:
            #time.sleep((START)*60)
            samples = (AMPLITUDE * np.sin(2 * np.pi * np.arange(SAMPLE_RATE * DURATION*60) * FREQUENCY / SAMPLE_RATE)).astype(np.float32)
            #tone = np.stack([samples, np.zeros(samples.shape, dtype=np.float32)], axis=1).astype(np.float32).reshape(-1)
            output_bytes = (1.0 * samples).tobytes()
            stream.write(output_bytes)
            INITVBR_EVT.set()
        if not TALK_EVT.is_set():
            TALK_EVT.set()
        while not END_EVT.wait(duration):
            if STOP_EVT.is_set():
                print("Stop talking...")
                break
            try:
                if dishNum == 1:
                    amp = AUDIO_Q1.get(block=False)
                else:
                    amp = AUDIO_Q2.get(block=False)
            except Empty:
                #continue
                amp = 0.0
            #samples = (librosa.tone(FREQUENCY, sr=SAMPLE_RATE, length=int(SAMPLE_RATE * duration))).astype(np.float32)
            samples = (AMPLITUDE * amp * np.sin(2 * np.pi * np.arange(SAMPLE_RATE * frameFPS) * FREQUENCY / SAMPLE_RATE)).astype(np.float32)
            #if dishNum == 2:
            #    tone = np.stack([samples, np.zeros(samples.shape, dtype=np.float32)], axis=1).astype(np.float32).reshape(-1) # left channel is silent
            #else:
            #    tone = np.stack([samples, np.zeros(samples.shape, dtype=np.float32)], axis=1).astype(np.float32).reshape(-1)

            output_bytes = samples.tobytes()
            stream.write(output_bytes)
            wavefile.writeframes(output_bytes)

    except KeyboardInterrupt:
        #pass
        if not END_EVT.is_set():
            END_EVT.set()
    finally:
        print("Closing audio streams...")
        stream.close()
        p.terminate()
        wavefile.close()
        #if not END_EVT.is_set():
        #    END_EVT.set()

parser = ArgumentParser()
parser.add_argument('-e', dest='exp', type=int, default=1)
parser.add_argument('-f', dest='frequency', type=int, default=300)
args = parser.parse_args()

if False:
    print(f"number of cpus: {multiprocessing.cpu_count()}")
    print(get_available_audio_devices())
    print(get_available_cameras())

#sub.call(f"rm -rf data{args.exp}", shell=True)
#sub.call(f"mkdir /data{args.exp}", shell=True)

cameras = ['http://10.243.106.67:8080/video', 'http://10.243.29.24:8080/video'] # dish 1: right, dish 2: left
speakers = [0, 1] # right, left

DEBUG = False
START = 5
DURATION = 5
TALK = 10
STOP = 5
SHOW = True # show the objects trajectory
SAMPLE_RATE = 48000 # in Hz, for the speakers

AMPLITUDE = 1.0
FREQUENCY = args.frequency  # sine frequency, Hz
plot_pause = 0.00001 # need to pause for dynamically plotting the amplitude of the sound
avg_windowSize = 30 # for exp3_f500: 100 ,averaging the straightness over the last 50 frames
variable_windowSize = True
THRESHOLD = 0.5 # in [0.0, 1.0], if the straightness is above this threshold, the sound is played
VELOCITY_WINDOW = 30
HEADING_WINDOW1 = 10
HEADING_WINDOW2 = 10
EXP = args.exp
duration = 1/5 # for exp3_f500: 1/25

frameWidth = 1920
frameHeight = 1080
frameFPS = 0.5

# tracker parameters
offset_x1 = -75
offset_y1 = -60
max_dist1 = 620
canny_thres1 = 150
nms_thres1 = 0.3

offset_x2 = -180
offset_y2 = -25
max_dist2 = 630
canny_thres2 = 100
nms_thres2 = 0.3

threads = [Process(target=capture, args=(1, cameras, duration, DEBUG)),\
           Process(target=capture, args=(2, cameras, duration, DEBUG)),\
           Process(target=track_display_record, args=(EXP, 1, frameWidth, frameHeight, frameFPS, duration, offset_x1, offset_y1, max_dist1, canny_thres1, nms_thres1, avg_windowSize, HEADING_WINDOW1, DEBUG)),
           Process(target=track_display_record, args=(EXP, 2, frameWidth, frameHeight, frameFPS, duration, offset_x2, offset_y2, max_dist2, canny_thres2, nms_thres2, avg_windowSize, HEADING_WINDOW2, DEBUG)),\
           Process(target=audio, args=(EXP, 1, SAMPLE_RATE, AMPLITUDE, FREQUENCY, frameFPS, speakers, duration, avg_windowSize, DEBUG)),\
           Process(target=audio, args=(EXP, 2, SAMPLE_RATE, AMPLITUDE, FREQUENCY, frameFPS, speakers, duration, avg_windowSize, DEBUG))]

queues = [AUDIO_Q1, AUDIO_Q2, DISP_Q1, DISP_Q2]

for thr in threads:
    thr.start()

startTime = datetime.now()
print(f"main thread: {startTime}")
# ploting the amplitude of sent vibration
#plt.ion()
#fig = plt.figure()
#ax = fig.add_subplot(1, 1, 1)
#plt.ylim((0, 1.5))
amplitudes1 = []
amplitudes2 = []
try:
    while not END_EVT.is_set():
        try:
            if (not START_EVT.is_set()) and ((datetime.now() - startTime).total_seconds() >= (START * 60)):
                START_EVT.set()
            if ((datetime.now() - startTime).total_seconds() > ((START+DURATION+TALK) * 60)):
                if not STOP_EVT.is_set():
                    STOP_EVT.set()
            if ((datetime.now() - startTime).total_seconds() > ((START+DURATION+TALK+STOP) * 60)):
                print("Stopping main thread...")
                END_EVT.set()
                break
            time.sleep(0.001)
            #amp1 = AMP_Q1.get(block=False)
            #amplitudes1.append(amp1)
            #amp2 = AMP_Q2.get(block=False)
            #amplitudes2.append(amp2)

            #plt.plot(amplitudes1, color='royalblue', linewidth=2)
            #plt.plot(amplitudes2, color='royalblue', linewidth=2)
            #plt.draw()
            #lt.ylim((0, 1.0))
            #plt.pause(plot_pause)
            #plt.clf()
        #except Empty:
        #    continue  
        except KeyboardInterrupt:
            break

finally:
    if not END_EVT.is_set():
        END_EVT.set()
    print("Empty queues...")
    for q in queues:
        while not q.empty():
            try:
                q.get(block=False)
            except (Empty, ValueError):
                break
        if hasattr(q, 'close'):
            q.close()

    print("Joining threads...")
    for thr in threads:
        thr.join()