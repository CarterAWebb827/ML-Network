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
import matplotlib.colors as clr
from sklearn.preprocessing import LabelEncoder

''' GLOBAL VARIABLES '''
dfList = []
labels = []

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

def concatData(mode=1, chunkSize=10000, maxMemory=0.85):
    projectDir = os.path.dirname(os.path.abspath(__file__))
    pathToCSV2017 = os.path.join(projectDir, 'CICIDS2017_improved')
    pathToCSV2018 = os.path.join(projectDir, 'CSECICIDS2018_improved')

    # Based on the mode, decide which dataset(s) to include
    csvCombined = []
    if mode == 0:
        # Load both datasets
        csv2017 = glob.glob(os.path.join(pathToCSV2017, "*.csv"))
        csv2018 = glob.glob(os.path.join(pathToCSV2018, "*.csv"))
        csvCombined = csv2017 + csv2018
    elif mode == 1:
        # Load only CICIDS2017 dataset
        csv2017 = glob.glob(os.path.join(pathToCSV2017, "*.csv"))
        csvCombined = csv2017
    elif mode == 2:
        # Load only CSECICIDS2018 dataset
        csv2018 = glob.glob(os.path.join(pathToCSV2018, "*.csv"))
        csvCombined = csv2018
    else:
        raise ValueError("Invalid mode selected. Choose 0 (both), 1 (CICIDS2017), or 2 (CSECICIDS2018).")

    # Iterate over files with tqdm for progress tracking
    for file in tqdm(csvCombined, desc="Reading CSV files"):
        # Read the CSV file into a DataFrame and append to the list
        chunkIteration = pd.read_csv(file, chunksize=chunkSize)
        for chunk in chunkIteration:
            if psutil.virtual_memory().percent >= (maxMemory * 100):
                # print("Memory limit has been reached, performing garbage collection...")
                gc.collect()

                time.sleep(2)
            
            dfList.append(chunk)

        # df = pd.read_csv(file)

    # Encode labels for each DataFrame
    aLabels = pd.concat([df['Label'] for df in dfList]).unique()
    le = LabelEncoder()
    le.fit(aLabels)
    for idx in range(len(dfList)):
        oLabels = dfList[idx]['Label'].unique()
        eLabels = le.transform(dfList[idx]['Label'])
        
        labelMap = {original: le.transform([original])[0] for original in oLabels}
        labels.append(labelMap)

        dfList[idx]['Label'] = eLabels


def plotData(dfList, columns, xlabel, ylabel):
    # Get a list of colors using matplotlib
    colors = list(clr.TABLEAU_COLORS)
    numColors = len(colors)

    # Number of DataFrames and columns to plot for each DataFrame
    numDFs = len(dfList)
    numColumns = len(columns)

    # Create a grid of subplots
    ncols = numColumns  # Each DataFrame will have numColumns plots in a row
    nrows = numDFs      # One row for each DataFrame

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18 * ncols, 8 * nrows))

    # If there's only one DataFrame, axes won't be a list, so we need to handle that case
    if nrows == 1:
        axes = [axes]

    # Loop through the DataFrames and create subplots
    for dfIdx, df in enumerate(dfList):
        for colIdx, col in enumerate(columns):
            ax = axes[dfIdx][colIdx] if nrows > 1 else axes[colIdx]  # Handle single-row or multi-row cases
            color = colors[(dfIdx + colIdx) % numColors]  # Cycle through colors

            # Plot the data
            ax.plot(df.index, df[col], label=f"{col} (Dataset {dfIdx + 1})", color=color, linestyle='-', linewidth=1)
            
            '''
            labelVal = df['Label'][dfIdx]

            if labelVal != 0 and labelVal in labels[dfIdx]:
                ax.plot(dfIdx, colIdx, 'ro', markersize=5)  # Mark the event with a red dot
                ax.annotate(f"{labels[dfIdx][labelVal]}", 
                            xy=(dfIdx, colIdx),
                            xytext=(dfIdx, colIdx + 0.5), 
                            fontsize=8,
                            arrowprops=dict(facecolor='black', arrowstyle='->'))
            '''

            # Set labels and title
            ax.set_title(f"{col} - Dataset {dfIdx + 1}", fontsize=12)
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            #ax.legend(loc="upper right")

            ax.grid(True, linestyle=':', linewidth=0.7, color='grey')
    
    plt.subplots_adjust(hspace=0.65, wspace=0.9, top=0.95, bottom=0.05, left=0.1, right=0.9)

    plt.show()

def main():
    # downloadData()

    mode = int(input("Enter dataset selection (0: Both, 1: CICIDS2017, 2: CSECICIDS2018): "))
    concatData(mode)

    # print(dfList[0].head())
    #print(labels)

    columns = ['Total Fwd Packet', 'Total Bwd packets', 'Average Packet Size']
    plotData(dfList, columns, "Time", "Number of Packets")

    # Clear large objects and force garbage collection
    # del dfList
    # gc.collect()  # Force garbage collection to free memory

if __name__ == "__main__":
    main()