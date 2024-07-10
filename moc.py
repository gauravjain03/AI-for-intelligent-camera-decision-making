import cv2

# Load pre-trained cascade classifier for object detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open video capture
cap = cv2.VideoCapture('your_video_source.mp4')  # replace 'your_video_source.mp4' with your actual video source

while cap.isOpened():
    # Read frame from the video
    ret, frame = cap.read()
    
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        # You can add additional logic here to identify suspicious conditions
        # For example, you could trigger an alarm or send a notification when a face is detected

    # Display the frame
    cv2.imshow('Suspicious Activity Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
