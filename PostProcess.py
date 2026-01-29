import cv2
import nms

class PostProcessor:

    def filter_result(self):
        boxes = self.result[:4, :]
        scores = self.result[5]
        mask = scores > 0.35    # boolean mask

        if mask.any():  # at least one score passes the threshold
            filtered_boxes = boxes[:, mask]
            filtered_scores = scores[mask]
            # Convert to list of tensors to mimic original behavior
            filtered_boxes = [filtered_boxes[:, i] for i in range(filtered_boxes.shape[1])]
            filtered_scores = filtered_scores.tolist()
        else:
            filtered_boxes = []
            filtered_scores = []

        return filtered_boxes, filtered_scores

    def draw_bbox(self, nms_boxes):
        for box in nms_boxes:
            box_np = box.cpu().numpy()
            x, y, w, h = box_np
            x1 = int((x - w / 2))
            y1 = int((y - h / 2))
            x2 = int(x1 + w)
            y2 = int(y1 + h)
            cv2.rectangle(self.image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    def put_inference_time(self):
        text = f"Inference: {self.time:.1f} ms"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        org = (10, 30)  # top-left corner
        color = (0, 255, 0)  # green (BGR)
        cv2.putText(self.image, text, org, font, font_scale, color, thickness)

    def set_image(self, image):
        self.image = image

    def set_result(self, result):
        self.result = result

    def set_time(self, time):
        self.time = time

    def get_frame(self):
        filtered_boxes, filtered_scores = self.filter_result()
        if len(filtered_boxes) > 0:
            distinct_indices = nms.non_max_suppression(filtered_boxes, filtered_scores, iou_threshold=0.5)
            nms_boxes = [filtered_boxes[i] for i in distinct_indices]
        else:
            nms_boxes = []
        self.draw_bbox(nms_boxes)
        self.put_inference_time()
        return self.image