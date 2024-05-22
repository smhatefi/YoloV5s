from models.yolov5s import YOLOv5s, load_official_weights
from utils.test import Tester

def main():
    # Instantiate the model
    model = YOLOv5s(num_classes=80)
    
    # Load the official weights
    model = load_official_weights(model, 'yolov5s.pt')

    # Create a tester instance
    tester = Tester(model, device='cuda')  # Change to 'cuda' if using GPU

    # Test on a custom image
    img_path = 'images/test1.jpg'
    boxes, scores, class_ids = tester.predict(img_path)

    # Define COCO class names
    class_names = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'TV', 'laptop', 'mouse', 'remote', 
        'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    # Visualize predictions
    tester.visualize_predictions(img_path, boxes, scores, class_ids, class_names)

if __name__ == "__main__":
    main()
