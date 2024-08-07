# import sys,os
# from satelliteDetection.logger import logging
# from satelliteDetection.exception import AppException
from satelliteDetection.pipeline.training_pipeline import TrainPipeline

obj = TrainPipeline()
obj.run_pipeline()