import cv2
import os
from detect import detect_objects

def image_test(image_path):
    if os.path.isdir(image_path):
        for filename in os.listdir(image_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                process_single_image(os.path.join(image_path, filename))
    else:
        process_single_image(image_path)

def process_single_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Invalid image file {image_path}")
        return

    detections = detect_objects(image)
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = f"{det['damage_type']}: {det['confidence']:.2f}"
        cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Image Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        image_test(sys.argv[1])
    else:
        print("Please provide an image file or directory path")