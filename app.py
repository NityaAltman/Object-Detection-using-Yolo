import sys,os
from satelliteDetection.pipeline.training_pipeline import TrainPipeline
from satelliteDetection.utils.main_utils import decodeImage, encodeImageIntoBase64
from flask import Flask, request, jsonify, render_template,Response
from flask_cors import CORS, cross_origin
from satelliteDetection.constant.application import APP_HOST, APP_PORT
from satelliteDetection.pipeline.training_pipeline import TrainPipeline
import json
import glob
import shutil

# obj = TrainPipeline()
# obj.start_model_trainer()
# obj.run_pipeline()

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
@app.route("/train")
def trainRoute():
    obj = TrainPipeline()
    obj.start_model_trainer()
    return "Training Successfull!!"

@app.route("/")
def home():
    return render_template("index.html")



@app.route("/predict", methods=['POST','GET'])
@cross_origin()
def predictRoute():
    try:
        image = request.json['image']
        decodeImage(image, clApp.filename)

        os.system("cd yolov5/ && python detect.py --weights mymodelepoch50.pt --img 416 --conf 0.5 --source ../data/inputImage.jpg")

        # Assuming detect.py outputs results in a JSON format saved to a file (e.g., results.json)
        with open('data/detection_results.json', 'r', encoding='utf-8') as f:
            detection_results = json.load(f)
        print(type(detection_results))  # Check the type of detection_results
        print(detection_results)  # Print the content of detection_results

        # Process detection results based on the actual structure of detection_results
        counts_result = {}
        if isinstance(detection_results, list):  # Example: handling a list of strings scenario
            for item in detection_results:
                key, value = item.split(":")
                counts_result[key.strip()] = int(value.strip())
        elif isinstance(detection_results, dict):  # If already a dictionary
            counts_result = detection_results
        elif isinstance(detection_results, str):  # If it's a string that needs parsing
            import ast
            counts_result = ast.literal_eval(detection_results)

        # Find the latest 'exp*' directory
        exp_dirs = glob.glob("yolov5/runs/detect/exp*")
        latest_exp_dir = max(exp_dirs, key=os.path.getctime) if exp_dirs else None

        if latest_exp_dir:
            image_path = os.path.join(latest_exp_dir, "inputImage.jpg")
            if not os.path.exists(image_path):
                return Response("Output image not found", status=500)

        opencodedbase64 = encodeImageIntoBase64(image_path)
        encoded_image = opencodedbase64.decode('utf-8')

        # Path to the directory containing the exp directories
        exp_dirs = glob.glob("yolov5/runs/detect/exp*")

        # Iterate over each directory and delete it
        for exp_dir in exp_dirs:
            shutil.rmtree(exp_dir, ignore_errors=True)

    except ValueError as val:
        print(val)
        return Response("Value not found inside  json data")
    except KeyError:
        return Response("Key value error incorrect key passed")
    except Exception as e:
        print(e)
        result = "Invalid input"

    return jsonify({"image": encoded_image, "counts": counts_result})

if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host=APP_HOST, port=APP_PORT)