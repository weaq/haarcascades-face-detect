import cv2, time

# Load cascade
# Multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Capture video from webcam.
cap = cv2.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height


while True:
    # Capture frame by frame
    ret , frame = cap.read()

    # if end source
    if not ret:
        break

    # Original fram
    img = frame.copy()

    # Convert grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=6,
            minSize=(20, 20)
            )

    # Draw rectangle around face
    for (x, y, w, h) in faces:
        roi_color = img[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Show image
    cv2.imshow('frame', frame)

    # Wait for keyboard press
    k = cv2.waitKey(1) & 0xff
     # Press q to exit
    if k==ord('q'):
        break
    # Press s to save
    if k == ord('s'):
        timestr = time.strftime("%Y%m%d%H%M%S")
        # Write capture to file
        cv2.imwrite('img/' + timestr + '.jpg',roi_color)

# Release capture
cap.release()
cv2.destroyAllWindows()
