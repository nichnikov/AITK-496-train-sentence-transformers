import os
import json
import logging
from src.data_types import Parameters


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', )

logger = logging.getLogger()
logger.setLevel(logging.INFO)

with open(os.path.join(os.getcwd(), "data", "config.json"), "r") as jf:
    config_dict = json.load(jf)

parameters = Parameters.parse_obj(config_dict)
print(parameters)