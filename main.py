import cv2

def draw_cross(frame, center, size=10, color=(0, 255, 0), thickness=2):
    """Draws a cross at the specified center coordinates."""
    x, y = center
    cv2.line(frame, (x - size, y), (x + size, y), color, thickness)
    cv2.line(frame, (x, y - size), (x, y + size), color, thickness)

def main(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read video frame.")
        return

    # Let the user select the region to track
    bbox = cv2.selectROI("Select Region", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Region")

    # Initialize the tracker
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, bbox)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        # Update the tracker
        success, bbox = tracker.update(frame)

        if success:
            # Draw a cross at the center of the bounding box
            x, y, w, h = map(int, bbox)
            center = (x + w // 2, y + h // 2)
            print("Tracker Coords: ", (x, y))
            draw_cross(frame, center)

        # Display the result
        cv2.imshow("Tracking", frame)

        # Exit on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "input_video.mp4"  # Replace with the path to your video file
    main(video_path)
