import cv2

# Global variables
roi_points = []
drawing = True

def click_event(event, x, y, flags, param):
    global roi_points, drawing

    if event == cv2.EVENT_LBUTTONDOWN and drawing:
        roi_points.append((x, y))
        # Draw a small circle on each click
        cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
        # Draw lines between points
        if len(roi_points) > 1:
            cv2.line(img, roi_points[-2], roi_points[-1], (255, 0, 0), 2)
        cv2.imshow("Image - Draw ROI (Press Enter to finish)", img)

def main(image_path):
    global img, roi_points, drawing

    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image from {image_path}")
        return

    clone = img.copy()
    cv2.namedWindow("Image - Draw ROI (Press Enter to finish)")
    cv2.setMouseCallback("Image - Draw ROI (Press Enter to finish)", click_event)

    print("Instructions:")
    print("- Click points on the image to draw the ROI polygon.")
    print("- Press Enter when done to finish and output coordinates.")
    print("- Press 'r' to reset points.")
    print("- Press 'q' to quit without saving.")

    while True:
        cv2.imshow("Image - Draw ROI (Press Enter to finish)", img)
        key = cv2.waitKey(1) & 0xFF

        if key == 13:  # Enter key to finish
            if len(roi_points) > 2:
                # Draw line from last to first point to close polygon
                cv2.line(img, roi_points[-1], roi_points[0], (255, 0, 0), 2)
                cv2.imshow("Image - Draw ROI (Press Enter to finish)", img)
                print("Final ROI points:")
                print(roi_points)
                cv2.waitKey(0)
                break
            else:
                print("Need at least 3 points to form a polygon.")
        elif key == ord('r'):  # Reset
            roi_points = []
            img = clone.copy()
            cv2.imshow("Image - Draw ROI (Press Enter to finish)", img)
            print("Reset ROI points.")
        elif key == ord('q'):  # Quit
            print("Quitting without saving.")
            roi_points = []
            break

    cv2.destroyAllWindows()
    return roi_points


if __name__ == "__main__":
    image_path = "G:\\desktop\\MyProjects\\DL_projects\\BPRD_usecase\\inference\\in.png"  # Replace with your image path
    points = main(image_path)
    print("Returned ROI points:", points)