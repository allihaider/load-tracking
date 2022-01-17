# Usage
The detection script requires 4 arguments:

* The path to input video to detect 

* The path to model to be used for detection 

* The number of classes detected by the model 

* The detection threshold 

Example:

	``` python3 detect_vid.py --input ../video/4m40s_C1595.mp4 --model ../model/model-c1595-17-19-11.pth --classes 21 --threshold 0.4 ```
