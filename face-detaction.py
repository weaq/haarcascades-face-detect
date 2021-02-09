import cv2

# Load cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# capture video from webcam.
cap = cv2.VideoCapture(0)

while True:
    # Capture frame by frame
    ret , frame = cap.read()

    # if end source
    if not ret:
        break

    # Convert grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangle around face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Show image
    cv2.imshow('frame', frame)

    # Press q to exit
    k = cv2.waitKey(1) & 0xff
    if k==ord('q'):
        break

# Release capture
cap.release()
cv2.destroyAllWindows()
