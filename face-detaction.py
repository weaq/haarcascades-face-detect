import cv2
import time
later = time.time()
# Load cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Capture video from webcam.
cap = cv2.VideoCapture(0)

while True:
    # Capture frame by frame
    ret , frame = cap.read()

    # If end source
    if not ret:
        break

    now = time.time()

    # Convert grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangle around face if has eyes
    for (x, y, w, h) in faces:

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:

            frame = cv2.rectangle(frame, (x, y), (x+w+3, y+h+3), (255, 0, 0), 2)

            if int(now - later) > 3 :
                later = now
                print(x, y)

    # Show image
    cv2.imshow('frame', frame)

    # Press q to exit
    k = cv2.waitKey(1) & 0xff
    if k==ord('q'):
        break

# Release capture
cap.release()
cv2.destroyAllWindows()
