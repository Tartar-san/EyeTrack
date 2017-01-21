import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def transformCoordinates(original, relative):
    """
    All locations in format (x, y, w, h)
    """
    return (original[0]+relative[0], original[1]+relative[1], relative[2], relative[3])

def detect_eyes(face, image):
    global eye_cascade
    (x, y, w ,h) = face
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_gray = gray[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(
        face_gray,
        minNeighbors = 10,
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    for i in range(len(eyes)):
        eyes[i] = transformCoordinates(face, eyes[i])
    return eyes

def detect_face(image):
    global face_cascade
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 10,
        minSize = (30, 30),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE | cv2.cv.CV_HAAR_FIND_BIGGEST_OBJECT
    )
    if len(faces) == 0:
        return False
    return faces[0]

def eyes_validator(eyes, image):
    return len(eyes) == 2

class PupilsDetector:

    def __init__(self, eyes):
        self.eyes = eyes
        self.detect()

    def detect(self, image):
        for (x, y, w, h) in self.eyes:
            eye = image[y:y+h,x:x+w]
            gauss = cv2.GaussianBlur(eye, (5, 5), 0)
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(gauss)
            thresh = cv2.threshold(gauss, minVal * 1.2, 255, cv2.THRESH_BINARY_INV)[1]
            im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            contour = None
            for c in contours:
                if cv2.pointPolygonTest(c, minLoc, False) >= 0:
                    contour = c
                    break

            if contour is None:
                self.pupils = (x+minLoc[0], y+minLoc[1], 1, 1)

            M = cv2.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            self.pupils = (x+cX, y+cY, 1, 1)
            self.eyes = (x+cX-self.eyes[2], y+cY-self.eyes[3], self.eyes[2], self.eyes[3])


    def pupils_location(self):
        return self.pupils

    def eyes_location(self):
        return self.eyes

class Main:

    def __init__(self):
        self.video_capture = cv2.VideoCapture(0)

    def painteyes(self, eyes, frame):
        for (x, y, w, h) in eyes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    def paintpupils(self, pupils, frame):
        pass

    def paintface(self, face, frame):
        for (x, y, w, h) in face:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    def calibration(self):
        eyes = []

        while eyes_validator(eyes):
            ret, frame = self.video_capture.read()

            if ret == 1:
                face = detect_face(frame)
                if face is not False:
                    eyes = detect_eyes(face, frame)

                self.painteyes(eyes, frame)

                cv2.imshow("Video", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.tracker = PupilsDetector(eyes)
        self.paintpupils(self.tracker.pupils_location(), frame)

    def run(self):
        self.calibration()

        while True:
            ret, frame = self.video_capture.read()

            if ret == 1:
                self.tracker.detect(frame)
                self.painteyes(self.tracker.eyes_location(), frame)
                self.paintpupils(self.tracker.pupils_location(), frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break





