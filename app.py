import flask
import configparser
import json
import os
from DataPreperation.FeaturePrep import consistent_col, get_codes
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

    config = configparser.ConfigParser()

    config["csvPaths"] = {"originalCSV": csv_path}
    config["modelData"] = {"predictionColumn": column_name}

    project_path = os.path.join(folder_location,
                                project_name)

    os.mkdir(project_path)

    dataframe = pd.read_csv(filepath_or_buffer=csv_path)

    for column in list(dataframe.columns):
        dataframe = consistent_col(dataframe, column)

    cat_col = list(dataframe.select_dtypes('category').columns)

    code_dict = {}

    for col in cat_col:
        file_name = "category_" + col + "_column" + ".json"
        file_path = os.path.join(folder_location,
                                 project_name,
                                 file_name)

        data_dict = get_codes(dataframe[col])

        code_dict["category_" + col + "_column"] = file_path

        with open(file_path, "w") as file:
            json.dump(data_dict, file)

        dataframe[col] = dataframe[col].cat.codes

    new_frame_path = os.path.join(folder_location,
                                  project_name,
                                  "CleanedData.csv")

    dataframe.to_csv(path_or_buf=new_frame_path,
                     index=False)

    config["codePaths"] = code_dict
    config["csvPaths"].update({"new_frame": new_frame_path})

    config_path = os.path.join(folder_location,
                               project_name,
                               "config.ini")

    with open(config_path, 'w') as configfile:
        config.write(configfile)

    status_code = flask.Response(status=201)
    return status_code
