import cv2

def test_webcam():
    # Create a VideoCapture object. 0 refers to the default webcam.
    # If you have multiple webcams, you might need to try 1, 2, etc.
    cap = cv2.VideoCapture("http://192.168.0.180:5000/stream?src=0")

    # Check if the webcam is opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam opened successfully. Press 'q' to quit.")

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        # If the frame was not read successfully, break the loop
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Display the captured frame
        cv2.imshow('Webcam Feed', frame)

        # Wait for a key press. If 'q' is pressed, exit the loop.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture object and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_webcam()