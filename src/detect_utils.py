import cv2
import numpy as np
import torchvision.transforms as transforms
from coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names

COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

transform = transforms.Compose([
	transforms.ToTensor(),
])

def predict(image, model, device, detection_threshold):
	
	image = transform(image).to(device)
	image = image.unsqueeze(0)
	outputs = model(image)

	pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
	pred_scores = outputs[0]['scores'].detach().cpu().numpy()
	pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
	pred_labels = outputs[0]['labels'].cpu().numpy()
	
	boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)
	classes = np.array(pred_classes)[pred_scores >= detection_threshold]
	labels = pred_labels[pred_scores >= detection_threshold]
	scores = pred_scores[pred_scores >= detection_threshold]
	
	return boxes, classes, labels, scores

def draw_boxes(boxes, classes, labels, scores, image):

	image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
	
	for i, box in enumerate(boxes):
		color = COLORS[labels[i]]
		cv2.rectangle(
			image,
			(int(box[0]), int(box[1])),
			(int(box[2]), int(box[3])),
			color, 2
		)
		cv2.putText(image, classes[i] + " " + str(scores[i]).split(".")[1][0:2], (int(box[0]), int(box[1]-5)),
					cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, 
					lineType=cv2.LINE_AA)

	return image
