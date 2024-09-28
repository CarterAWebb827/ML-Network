import os                           # Allows the use of os commands
import zipfile                      # Enables the use of zip
import subprocess                   # Enables the ability of subprocessing
import shutil                       # Utility functions for copying and archiving files

import numpy as np                  # Library for numerical computations
import pandas as pd                 # Library for data maniplulation and analysis
import matplotlib.pyplot as plt     # Library for plotting data

def downloadData():
    # Ask the user for the path to their kaggle.json file
    kaggleKeyPath = input("Please provide the full path to your kaggle.json file: ")

    # Check if the provided path exists
    if not os.path.exists(kaggleKeyPath):
        raise FileNotFoundError(f"File not found at {kaggleKeyPath}")

    # Create the .kaggle directory if it doesn't exist
    kaggleDir = os.path.expanduser("~/.kaggle")
    if not os.path.exists(kaggleDir):
        os.makedirs(kaggleDir)

    # Copy the kaggle.json file to the correct location
    kaggleKeyDest = os.path.join(kaggleDir, "kaggle.json")
    shutil.copyfile(kaggleKeyPath, kaggleKeyDest)

    # Set the permissions of the kaggle.json file (works on Unix-based systems; Windows doesn't need this)
    os.chmod(kaggleKeyDest, 0o600)

    # Check if the datasets already exists before downloading
    dataset2017 = "CICIDS2017_improved"
    dataset2018 = "CSECICIDS2018_improved"
    if not os.path.exists(dataset2017) or not os.path.exists(dataset2018):
        # Install Kaggle API using python -m pip
        subprocess.run(["python", "-m", "pip", "install", "kaggle"], check=True)

        # Download the dataset using the Kaggle API
        dataset = "ernie55ernie/improved-cicids2017-and-csecicids2018"
        subprocess.run(["kaggle", "datasets", "download", "-d", dataset], check=True)

        # Unzip the downloaded dataset
        zipFile = "improved-cicids2017-and-csecicids2018.zip"
        with zipfile.ZipFile(zipFile, 'r') as zipRef:
            print("File is being extracted")
            zipRef.extractall()

        # Delete the zip file after extraction
        os.remove(zipFile)
        print(f"Dataset downloaded, extracted, and zip file {zipFile} deleted.")
    else:
        print(f"Dataset already exists in the {dataset2017} and {dataset2018} folders. No download needed.")

def main():
    downloadData()

if __name__ == "__main__":
    main()