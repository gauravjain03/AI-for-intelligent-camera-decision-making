import cv2
import numpy as np

def detect_metal(frame):
    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the color of the metal
    lower_bound = np.array([0, 0, 100])  # Adjust these values based on your specific case
    upper_bound = np.array([180, 255, 255])

    # Create a mask to filter out everything except the metal color
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours (metallic objects) are found
    metal_detected = len(contours) > 0

    # Draw bounding boxes around detected metallic objects
    if metal_detected:
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame, metal_detected

def main():
    # Open the video capture (0 corresponds to the default camera)
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Check if the frame is successfully captured
        if not ret:
            break

        # Detect metal in the frame
        frame, metal_detected = detect_metal(frame)

        # Display the resulting frame
        cv2.imshow('Metal Detection', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
