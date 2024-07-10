import cv2

# Load pre-trained cascade classifier for object detection
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

# Open video capture
cap = cv2.VideoCapture(0)  # Use 0 for default camera, or provide the camera index (e.g., 1, 2) for additional cameras

while cap.isOpened():
    # Read frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale for body detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect bodies in the frame
    bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected bodies
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Calculate the center of the bounding box
        center_x = x + w // 2
        center_y = y + h // 2

        # Display the center coordinates
        cv2.putText(frame, f'Center: ({center_x}, {center_y})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with centralized suspicious conditions
    cv2.imshow('Suspicious Condition Centralization', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

