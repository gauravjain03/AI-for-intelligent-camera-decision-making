import cv2

# Open video capture
cap = cv2.VideoCapture(0)  # Use 0 for default camera, or provide the camera index (e.g., 1, 2) for additional cameras

# Initialize background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

while cap.isOpened():
    # Read frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Apply morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    # Find contours in the foreground mask
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around moving objects (persons)
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # You can adjust the area threshold based on your needs
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the frame with movement recognition
    cv2.imshow('Movement Recognition', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

