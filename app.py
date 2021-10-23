import flask
import configparser
import json
import os
from DataPreperation.FeaturePrep import convert_columns, convert_categories
from flask import Flask, request
import pandas as pd


app = Flask(__name__)


@app.route("/init_data/", methods=["POST"])
def clean_data():
    data = json.loads(request.data)

    column_name = data["columnName"]
    csv_path = data["csvPath"]
    folder_location = data["folderLocation"]
    project_name = data["projectName"]

    project_path = os.path.join(folder_location,
                                project_name)
    os.mkdir(project_path)

    dataframe = pd.read_csv(filepath_or_buffer=csv_path)
    dataframe = convert_columns(dataframe)

    config_path = os.path.join(folder_location,
                               project_name,
                               "config.ini")

    config = configparser.ConfigParser()

    config["csvPaths"] = {"originalCSV": csv_path}
    config["modelData"] = {"predictionColumn": column_name}

    with open(config_path, 'w') as configfile:
        config.write(configfile)

    status_code = flask.Response(status=201)
    return status_code
