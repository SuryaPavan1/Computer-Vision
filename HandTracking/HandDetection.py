import cv2
import mediapipe as mp


class HandDetector():
    def __init__(self, mode=False, maxHands=2, minDetections=0.5, minTracking=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.minDetections = minDetections
        self.minTracking = minTracking
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils

    def FindHands(self, frame, draw=True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        if results.multi_hand_landmarks:
            for handLoc in results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLoc, self.mpHands.HAND_CONNECTIONS)
        return frame

    def FindHandLocation(self, frame, handNumber=0, draw=True):
        locList = []
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        if results.multi_hand_landmarks:
            myHand = results.multi_hand_landmarks[handNumber]
            for Id, loc in enumerate(myHand.landmark):
                h, w, c = frame.shape
                cx, cy = int(loc.x * w), int(loc.y * h)
                locList.append([Id, cx, cy])
            if draw:
                self.mpDraw.draw_landmarks(frame, myHand, self.mpHands.HAND_CONNECTIONS)

        return locList
