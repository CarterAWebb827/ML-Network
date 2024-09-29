''' GENERAL LIBRARIES '''
import os                           # Allows the use of os commands
import zipfile                      # Enables the use of zip
import subprocess                   # Enables the ability of subprocessing
import shutil                       # Utility functions for copying and archiving files
import glob                         
from tqdm import tqdm               # Tool to add progress bars
import psutil
import time
import gc

''' MACHINE LEARNING LIBRARIES'''
import numpy as np                  # Library for numerical computations
import pandas as pd                 # Library for data maniplulation and analysis
import matplotlib.pyplot as plt     # Library for plotting data
from sklearn.preprocessing import LabelEncoder

''' GLOBAL VARIABLES '''
# RAMLIMITGB = 24.0
# RAMLIMBYTE = RAMLIMITGB * 1024 * 1024 * 1024

dfList = []

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

def concatData():
    projectDir = os.path.dirname(os.path.abspath(__file__))
    pathToCSV2017 = os.path.join(projectDir, 'CICIDS2017_improved')
    pathToCSV2018 = os.path.join(projectDir, 'CSECICIDS2018_improved')

    csv2017 = glob.glob(os.path.join(pathToCSV2017, "*.csv"))
    csv2018 = glob.glob(os.path.join(pathToCSV2018, "*.csv"))
    csvCombined = csv2017 + csv2018

    # Iterate over files with tqdm for progress tracking
    for file in tqdm(csvCombined, desc="Reading CSV files"):
        # checkRAMLimit()

        # Read the CSV file into a DataFrame and append to the list
        df = pd.read_csv(file)
        dfList.append(df)

    # Optional: If you want to encode labels for each DataFrame
    le = LabelEncoder()
    for idx in range(len(dfList)):
        dfList[idx]['Label'] = le.fit_transform(dfList[idx]['Label'])

'''
def checkRAMLimit():
    ramInfo = psutil.virtual_memory()
    usedRam = ramInfo.used
    
    # If used memory exceeds the limit, pause the execution
    if usedRam > RAMLIMBYTE:
        print(f"Memory limit exceeded: {usedRam / (1024**3):.2f} GB used. Waiting for memory to be available...")
        while psutil.virtual_memory().used > RAMLIMBYTE:
            time.sleep(5)  # Wait 5 seconds before rechecking
'''

def main():
    # downloadData()

    concatData()

    print(dfList[0].head())

    # Clear large objects and force garbage collection
    del dfList
    gc.collect()  # Force garbage collection to free memory

if __name__ == "__main__":
    main()