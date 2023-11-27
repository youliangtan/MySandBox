import cv2
import mediapipe as mp
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from hazards import hazards
import mediapipe as mp
import cv2
# @markdown We implemented some functions to visualize the object detection results. <br/> Run the following cell to activate the functions.
import cv2
import numpy as np
from typing import Tuple

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


def visualize(
    image,
    detection_result
) -> Tuple[np.ndarray, dict]:
    """Draws bounding boxes on the input image and return it.
    Args:
      image: The input RGB image.
      detection_result: The list of all "Detection" entities to be visualize.
    Returns:
      Image with bounding boxes.
    """
    print('Number of detections: ', len(detection_result.detections))
    
    results = {}
    for detection in detection_result.detections:

        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        if category_name not in hazards:
            continue

        print(category_name)
        print(hazards[category_name])        
        results[category_name] = hazards[category_name]
        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 5)

        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (MARGIN + bbox.origin_x,
                         MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
    print("--------------------")
    return image, results

class HazardDetector:
    def __init__(self) -> None:
        base_options = python.BaseOptions(model_asset_path='efficientdet.tflite')
        options = vision.ObjectDetectorOptions(base_options=base_options,
                                            score_threshold=0.2,
                                            )
        self.detector = vision.ObjectDetector.create_from_options(options)

    def detect(self, image_path, viz=False):
        image = mp.Image.create_from_file(image_path)

        # STEP 4: Detect objects in the input image.
        detection_result = self.detector.detect(image)

        # STEP 5: Process the detection result. In this case, visualize it.
        image_copy = np.copy(image.numpy_view())
        annotated_image, results = visualize(image_copy, detection_result)
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        if viz:
            cv2.imshow('MediaPipe Objectron', annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return annotated_image, results


if __name__ == '__main__':
    detector = HazardDetector()
    annotated_image, results = detector.detect("scene/image2.jpg", viz=True)
    print(results)
