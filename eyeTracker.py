import cv2
import numpy as np
import math

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def transformCoordinates(original, relative):
    """
    All locations in format (x, y, w, h)
    """
    return (original[0]+relative[0], original[1]+relative[1], relative[2], relative[3])

def detect_eyes(face, gray):
    global eye_cascade
    (x, y, w ,h) = face
    face_gray = gray[y:y+h//2, x:x+w]
    eyes = eye_cascade.detectMultiScale(
        face_gray,
        minNeighbors = 10,
        flags = cv2.CASCADE_SCALE_IMAGE
        # cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    for i in range(len(eyes)):
        eyes[i] = transformCoordinates(face, eyes[i])
    return eyes

def detect_face(gray):
    global face_cascade
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 10,
        minSize = (30, 30),
        flags = cv2.CASCADE_DO_ROUGH_SEARCH
        # cv2.cv.CV_HAAR_SCALE_IMAGE | cv2.cv.CV_HAAR_FIND_BIGGEST_OBJECT
    )
    if len(faces) == 0:
        return False
    return faces[0]

def eyes_validator(eyes):
    if len(eyes) != 2:
        return False

    if float(eyes[0]['r']) == 0 or float(eyes[1]['r']) == 0:
        print 'radius'
        return False

    if float(eyes[0]['r'])/float(eyes[1]['r']) > 2.5:
        print 'ratio'
        return False

    averRadius = (float(eyes[0]['r']) + float(eyes[1]['r']))/2
    dist = math.sqrt((eyes[0]['x'] - eyes[1]['x'])**2 + (eyes[0]['y'] - eyes[1]['y'])**2)

    if dist/averRadius > 25 or dist/averRadius < 3:
        print "dist " + str(dist/averRadius)
        return False

    return True


class PupilsDetector:

    def __init__(self):
        self.eyes = None

    def find(self, image, eyes):
        self.eyes = []
        for (x, y, w, h) in eyes:
            eye = image[y:y+h, x:x+w]
            if eye is None:
                return False
            eye = self.detect(eye)
            eye['x'] += x
            eye['y'] += y
            self.eyes.append(eye)
        return eyes_validator(self.eyes)

    def track(self, image, previous, frame = None):
        for i, eye in enumerate(self.eyes):
            x1 = max(0, eye['x']-max(5*eye['r'], 10))
            x2 = min(eye['x']+max(5*eye['r'], 10), image.shape[1]-1)
            y1 = max(0, eye['y']-max(3*eye['r'], 6))
            y2 =  min(eye['y']+max(3*eye['r'], 6), image.shape[0]-1)
            eye = image[y1:y2, x1:x2]
            #prev = np.copy(previous[y1:y2, x1:x2])

            #frameDelta = cv2.absdiff(prev, eye)
            #thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
            #cv2.imshow("diff", thresh)

            if frame is not None:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            eye = self.detect(eye)
            eye['x'] += x1
            eye['y'] += y1

            self.eyes[i] = eye

        return eyes_validator(self.eyes)

    def detect(self, eye):
        gauss = cv2.GaussianBlur(eye, (5, 5), 0)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(gauss)
        thresh = cv2.threshold(gauss, minVal * 1.2, 255, cv2.THRESH_BINARY_INV)[1]
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contourIm = np.zeros(thresh.shape, dtype="uint8");
        cv2.drawContours(contourIm, contours, -1, 255, 1)
        contour = None
        for c in contours:
            if cv2.pointPolygonTest(c, minLoc, False) >= 0:
                contour = c
                break

        if contour is None:
            #self.pupils = (minLoc[0], minLoc[1], 1, 1)
            cv2.circle(gauss, (minLoc[0], minLoc[1]), 2, (0, 0, 255), 3)
            cv2.imshow("gauss", gauss)
            cv2.imshow("eye", thresh)
            cv2.imshow("thresh", contourIm)
            return {'x': minLoc[0], 'y': minLoc[1], 'r': 0}

        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        radius = 0
        for point in contour:
            radius += math.sqrt(abs(cX - point[0][0]) ** 2 + abs(cY - point[0][1]) ** 2)
        radius /= len(contour)

        return {'x': cX, 'y': cY, 'r': int(radius), 'previous': gauss}

    #def pupils_location(self):
    #    return self.pupils

    def eyes_location(self):
        return self.eyes

    def paintpupils(self, frame):
        for eye in self.eyes:
            cv2.circle(frame, (eye['x'], eye['y']), 1, (0, 0, 255), 1)
            cv2.circle(frame, (eye['x'], eye['y']), eye['r'], (0, 0, 255), 1)

class Main:

    def __init__(self):
        self.video_capture = cv2.VideoCapture(0)
        self.tracker = PupilsDetector()
        self.previous = None

    def painteyes(self, eyes, frame):
        for (x, y, w, h) in eyes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    def paintface(self, face, frame):
        for (x, y, w, h) in face:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    def calibration(self):
        eyes = []
        ret, frame = self.video_capture.read()

        while True:
            ret, frame = self.video_capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if ret == 1:
                face = detect_face(gray)
                if face is not False:
                    eyes = detect_eyes(face, gray)

                    self.painteyes(eyes, frame)
                    (x, y, w, h) = face
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    if self.tracker.find(gray, eyes):
                        self.tracker.paintpupils(frame)
                        return True

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            self.previous = gray

            cv2.imshow("Video", frame)

        #self.paintpupils(self.tracker.pupils_location(), gray)

    def run(self):
        self.calibration()
        ret, frame = self.video_capture.read()
        while True:
            cv2.imshow("Video", frame)
            ret, frame = self.video_capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if ret == 1:
                if not self.tracker.track(gray, self.previous, frame):
                    self.calibration()
                self.tracker.paintpupils(frame)

            self.previousFrame = gray
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break





go = Main()
go.run()