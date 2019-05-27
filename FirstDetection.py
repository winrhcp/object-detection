from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

name = input()
name_image = name + ".jpg"
exeuted_image = name + "new.jpg"

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , name_image), output_image_path=os.path.join(execution_path , exeuted_image))

for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )

from PIL import Image
image = Image.open(exeuted_image)
image.show()