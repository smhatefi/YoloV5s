import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

class Tester:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def preprocess_image(self, img_path, img_size=640):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        img = torch.tensor(img, dtype=torch.float32).to(self.device)
        return img

    def postprocess_output(self, outputs, img_size=640, conf_threshold=0.25, iou_threshold=0.45):
        boxes, scores, class_ids = [], [], []
        for i in range(len(outputs)):
            output = outputs[i]
            grid_size = output.shape[2]
            stride = img_size / grid_size
            output = output.permute(0, 2, 3, 1).contiguous().view(-1, self.model.num_classes + 5)
            conf_mask = output[:, 4] > conf_threshold
            output = output[conf_mask]
            if not output.shape[0]:
                continue
            box_scores = output[:, 4]
            class_scores, class_ids_ = torch.max(output[:, 5:], dim=1)
            scores_ = box_scores * class_scores
            box_class_filter = scores_ > conf_threshold
            output = output[box_class_filter]
            scores_ = scores_[box_class_filter]
            class_ids_ = class_ids_[box_class_filter]

            boxes_ = output[:, :4]
            boxes_[:, 0] = (boxes_[:, 0] - boxes_[:, 2] / 2) * stride
            boxes_[:, 1] = (boxes_[:, 1] - boxes_[:, 3] / 2) * stride
            boxes_[:, 2] = (boxes_[:, 0] + boxes_[:, 2] / 2) * stride
            boxes_[:, 3] = (boxes_[:, 1] + boxes_[:, 3] / 2) * stride

            boxes.append(boxes_)
            scores.append(scores_)
            class_ids.append(class_ids_)

        if not boxes:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])

        boxes = torch.cat(boxes, dim=0)
        scores = torch.cat(scores, dim=0)
        class_ids = torch.cat(class_ids, dim=0)

        nms_indices = torch.ops.torchvision.nms(boxes, scores, iou_threshold)
        boxes, scores, class_ids = boxes[nms_indices], scores[nms_indices], class_ids[nms_indices]

        return boxes, scores, class_ids

    def predict(self, img_path, conf_threshold=0.25, iou_threshold=0.45):
        img = self.preprocess_image(img_path)
        with torch.no_grad():
            outputs = self.model(img)
        boxes, scores, class_ids = self.postprocess_output(outputs, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
        return boxes, scores, class_ids

    def visualize_predictions(self, img_path, boxes, scores, class_ids, class_names, img_size=640):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        plt.figure(figsize=(12, 12))
        plt.imshow(img)
        ax = plt.gca()
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box.int()
            label = f"{class_names[class_id]}: {score:.2f}"
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='green', linewidth=2)
            ax.add_patch(rect)
            ax.text(x1, y1, label, color='white', fontsize=12, bbox=dict(facecolor='green', alpha=0.5))
        plt.axis('off')
        plt.show()

# Example usage:
# from models.yolov5s import YOLOv5s, load_official_weights
# model = YOLOv5s(num_classes=80)
# model = load_official_weights(model, 'yolov5s.pt')
# tester = Tester(model, device='cuda')
# boxes, scores, class_ids = tester.predict('path/to/image.jpg')
# class_names = ['class1', 'class2', ..., 'class80']  # Replace with actual class names
# tester.visualize_predictions('path/to/image.jpg', boxes, scores, class_ids, class_names)
