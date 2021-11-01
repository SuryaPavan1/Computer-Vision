# import modules
import cv2
import time
import numpy as np
import HandDetection as hd
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

color = {'red': (0, 0, 255), 'green': (0, 255, 0), 'blue': (255, 0, 0)}


# Function to convert co-ordinate distances to volume ranges
def ConvertToVolume(minVol, maxVol, firstLoc, secondLoc):
    if len(firstLoc) == 2 and len(secondLoc) == 2:
        length = math.hypot(firstLoc[1] - secondLoc[1], firstLoc[0] - secondLoc[0])
        vol = np.interp(length, [35, 250], [minVol, maxVol])
        return vol
    return -1


def main():
    windowName = 'Video'
    pTime = 0
    cTime = 0
    width = 640
    height = 480
    start_point = (10, 100)
    end_point = (30, 400)
    cap = cv2.VideoCapture(0)
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    length =volume.GetMasterVolumeLevel()
    volrange = volume.GetVolumeRange() #gets volume range
    minVol = volrange[0]
    maxVol = volrange[1]
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(windowName, width, height)
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 2)
        handDetection = hd.HandDetector(minDetections=0.7)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        lmlist = handDetection.FindHandLocation(frame)
        if lmlist:
            center = [(lmlist[4][1] + lmlist[8][1]) // 2, (lmlist[4][2] + lmlist[8][2]) // 2]
            length = ConvertToVolume(minVol, maxVol, lmlist[4][1:], lmlist[8][1:])
            volume.SetMasterVolumeLevel(int(length), None)
            cv2.line(frame, lmlist[4][1:], lmlist[8][1:], color['blue'], 5)
            cv2.circle(frame, lmlist[4][1:], 15, color['blue'], -1)
            cv2.circle(frame, lmlist[8][1:], 15, color['blue'], -1)
            if length < minVol + 5:
                cv2.circle(frame, center, 15, color['green'], -1)
            elif length >= maxVol:
                cv2.circle(frame, center, 15, color['red'], -1)
            else:
                cv2.circle(frame, center, 15, color['blue'], -1)

        pTime = cTime
        cv2.putText(frame, "fps: " + str(int(fps)), (5, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.putText(frame, str(int((minVol - length)/ (0.01*minVol))) + "%", (10, 99), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
        a = np.interp(length, [minVol, maxVol], [400, 100])
        image = cv2.rectangle(frame, start_point, end_point, color['blue'], 1)
        image = cv2.rectangle(frame, (start_point[0], int(a)), end_point, color['blue'], -1)
        cv2.imshow(windowName, image)
        if cv2.waitKey(1) == 27:
            return


if __name__ == "__main__":
    main()
