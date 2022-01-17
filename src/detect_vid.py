import argparse
import torchvision
import torch
import cv2
import detect_utils

FRAME_COUNT = 0

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='path to input video')
parser.add_argument('-m', '--model', help='path to model')
parser.add_argument('-c', '--classes', help='number of classes')
parser.add_argument('-t', '--threshold', help='model threshold')
args = vars(parser.parse_args())

model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(num_classes=int(args['classes']), pretrained=False, min_size=800)
checkpoint = torch.load(f"{args['model']}")
model.load_state_dict(checkpoint['model'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.eval().to(device)

cap = cv2.VideoCapture(args['input'])
if (cap.isOpened() == False):
	print('Error while trying to read video. Please check path again')

while(cap.isOpened()):
	ret, frame = cap.read()

	if ret == True:
		FRAME_COUNT += 1

		if FRAME_COUNT % 30 != 0:
			continue
		
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame = cv2.resize(frame, (1280, 720), interpolation = cv2.INTER_AREA)

		with torch.no_grad():
			boxes, classes, labels, scores = detect_utils.predict(frame, model, device, float(args['threshold']))			
		
		# Add your code here 
		
		frame = detect_utils.draw_boxes(boxes, classes, labels, scores, frame)
		cv2.imshow('image', frame)
				
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	else:
		break
