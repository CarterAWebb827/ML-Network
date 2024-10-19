""" GENERAL LIBRARIES """
import os                           # Allows the use of os commands
import zipfile                      # Enables the use of zip
import subprocess                   # Enables the ability of subprocessing
import shutil                       # Utility functions for copying and archiving files
import glob                         
from tqdm import tqdm               # Tool to add progress bars
import gc
import re                           # For regular expression operations

""" MACHINE LEARNING LIBRARIES """
import numpy as np                  # Library for numerical computations
import pandas as pd                 # Library for data maniplulation and analysis
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # Library for plotting data
import matplotlib.colors as clr
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB # Naive Bayes
from sklearn.linear_model import LogisticRegression # Logistic Regression
import tensorflow as tf

""" GLOBAL VARIABLES """
dfList = []
labels = []

def downloadData():
    # Ask the user for the path to their kaggle.json file
    kaggleKeyPath = input("Please provide the full path to your kaggle.json file: ")

    # Check if the provided path exists
    if not os.path.exists(kaggleKeyPath):
        raise FileNotFoundError(f"File not found at {kaggleKeyPath}")

    # Create the .kaggle directory if it doesn"t exist
    kaggleDir = os.path.expanduser("~/.kaggle")
    if not os.path.exists(kaggleDir):
        os.makedirs(kaggleDir)

    # Copy the kaggle.json file to the correct location
    kaggleKeyDest = os.path.join(kaggleDir, "kaggle.json")
    shutil.copyfile(kaggleKeyPath, kaggleKeyDest)

    # Set the permissions of the kaggle.json file (works on Unix-based systems; Windows doesn"t need this)
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
        with zipfile.ZipFile(zipFile, "r") as zipRef:
            print("File is being extracted")
            zipRef.extractall()

        # Delete the zip file after extraction
        os.remove(zipFile)
        print(f"Dataset downloaded, extracted, and zip file {zipFile} deleted.")
    else:
        print(f"Dataset already exists in the {dataset2017} and {dataset2018} folders. No download needed.")

def concatData(mode=1, chunkSize=10000, maxMemory=0.85):
    global dfList
    global labels
    projectDir = os.path.dirname(os.path.abspath(__file__))
    pathToCSV2017 = os.path.join(projectDir, "CICIDS2017_improved")
    pathToCSV2018 = os.path.join(projectDir, "CSECICIDS2018_improved")

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
        df = pd.read_csv(file)
        dfList.append(df)

        """
        chunkIteration = pd.read_csv(file, chunksize=chunkSize)
        for chunk in chunkIteration:
            if psutil.virtual_memory().percent >= (maxMemory * 100):
                # print("Memory limit has been reached, performing garbage collection...")
                gc.collect()

                time.sleep(2)
            
            dfList.append(chunk)
        """

    # Encode labels for each DataFrame
    aLabels = pd.concat([df["Label"] for df in dfList]).unique()
    le = LabelEncoder()
    le.fit(aLabels)
    for idx in range(len(dfList)):
        oLabels = dfList[idx]["Label"].unique()
        eLabels = le.transform(dfList[idx]["Label"])
        
        labelMap = {original: le.transform([original])[0] for original in oLabels}
        labels.append(labelMap)

        #print (labelMap)

        dfList[idx]["Label"] = eLabels

def plotData(columns, xlabel, ylabel, sOn, labelMap):
    global dfList
    numColors = 27  # Example color cycle
    cmap = plt.colormaps.get_cmap("tab20")
    colors = cmap(np.linspace(0, 1, numColors))

    if not os.path.exists("Figures"):
        os.makedirs("Figures")

    if sOn:
        if not os.path.exists("Figures/SubPlots"):
            os.makedirs("Figures/SubPlots")
        # Loop through each DataFrame and each column
        for dfIdx, df in enumerate(dfList):
            df = df.dropna()  # Removes rows with NaN values
            for colIdx, col in enumerate(columns):
                # Create a new figure for each dataset and column
                fig, ax = plt.subplots(figsize=(10, 6))  # Adjust size if necessary

                # Plot the data
                ax.plot(df.index, df[col], label=f"{col} (Dataset {dfIdx + 1})", color=colors[(dfIdx + colIdx) % numColors], linestyle="-", linewidth=1)

                """
                labelVal = df["Label"][dfIdx]

                if labelVal != 0 and labelVal in labels[dfIdx]:
                    ax.plot(dfIdx, colIdx, "ro", markersize=5)  # Mark the event with a red dot
                    ax.annotate(f"{labels[dfIdx][labelVal]}", 
                                xy=(dfIdx, colIdx),
                                xytext=(dfIdx, colIdx + 0.5), 
                                fontsize=8,
                                arrowprops=dict(facecolor="black", arrowstyle="->"))
                """

                # Set labels and title for each individual plot
                ax.set_title(f"{col} - Dataset {dfIdx + 1}", fontsize=12)
                ax.set_xlabel(xlabel, fontsize=10)
                ax.set_ylabel(ylabel, fontsize=10)
                ax.grid(True, linestyle=":", linewidth=0.7, color="grey")

                # Adjust layout to avoid overlapping elements
                plt.tight_layout()

                # Save each plot with a unique filename
                plt.savefig(f"Figures/SubPlots/Dataset{dfIdx+1}_{col}.png")
                plt.close(fig)  # Close the figure after saving to avoid memory buildup
    
    if not os.path.exists("Figures/Histograms"):
            os.makedirs("Figures/Histograms")

    for dfIdx, df in enumerate(dfList):
        # Get numerical columns excluding "Label" and "id" or any non-numeric columns
        numericCols = df.select_dtypes(include=[np.number]).columns.tolist()
        numericCols.remove("Label")  # Remove Label

        for col in numericCols:
            plt.figure(figsize=(15, 15))

            # Initialize an empty list to hold average values for each label
            averages = []

            # Plot histogram for each label type
            for label in df["Label"].unique():
                # Get data for the current label
                data = df[df["Label"] == label][col]
                # Remove NaN and inf values from the data
                data = data[np.isfinite(data)]

                if len(data) > 0:  # Check if there is data to plot
                    avgVal = data.mean()

                    oLabel = next((orig for orig, enc in labelMap[dfIdx].items() if enc == label), str(label))

                    averages.append((oLabel, avgVal))
                    
                    # Use colormap to get color for the label
                    plt.hist(data, bins=30, alpha=0.5, color=colors[label], 
                            label=f"{oLabel} (Avg: {avgVal:.2f})", edgecolor="black")

            # Set titles and labels
            plt.title(f"Histogram of {col} for DataFrame {dfIdx + 1}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.legend(title="Label (Average Value)")
            
            # Show grid
            plt.grid(axis="y", alpha=0.75)

            if not os.path.exists(f"Figures/Histograms/{dfIdx + 1}"):
                os.makedirs(f"Figures/Histograms/{dfIdx + 1}")

            sCol = re.sub(r"[^\w\s]", "", col)  # Remove non-alphanumeric characters
            sCol = sCol.replace(" ", "_")  # Replace spaces with underscores

            plt.savefig(f"Figures/Histograms/{dfIdx + 1}/Hist{dfIdx + 1}_{sCol}.png")

            # Close the figure to free up memory
            plt.close()

def scaleDS(df, cToDrop, overSample=False):
    X = df.drop(columns=cToDrop)
    #X.drop(columns="Label", inplace=True)
    
    # Handle NaN values and Infinite values
    X = X[np.isfinite(X).all(axis=1)]
    y = df[df.columns[-2]][X.index].values  # Align y with the index of X after dropping rows

    X.dropna(inplace=True)
    if np.isinf(X).sum().sum() > 0:
        print("\nWarning: Infinite values still present after replacement.\n")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if isinstance(y, np.ndarray):
        y = pd.Series(y)

    if overSample:
        unique = y.unique()
        #print(f"Unique Classes in y: {unique}")

        if len(unique) >= 2:
            ros = RandomOverSampler()
            X, y = ros.fit_resample(X, y)
    
    if X.shape[0] != len(y):
        print(f"Dimension mismatch: X has {X.shape[0]} rows, y has {len(y)} entries.")

    data = np.hstack((X, np.reshape(y, (-1, 1))))

    return data, X, y

def nb(XSetArr, ySetArr, xTestArr, yTestArr):
    nbModel = GaussianNB()

    # Concatenate all training data
    XTrainCom = np.concatenate(XSetArr, axis=0)  # Combine all training features
    yTrainCom = np.concatenate(ySetArr, axis=0)  # Combine all training labels

    print("Fitting Naive Bayes Model...")
    # Fit the model using the combined training data
    nbModel.fit(XTrainCom, yTrainCom)
    
    print("Finished Fitting, Beginning Prediciton...")
    # Predict using the test sets
    yPred = []
    for i in range(len(xTestArr)):
        preds = nbModel.predict(xTestArr[i])
        yPred.append(preds)
    
    # Flatten yPred if you want a single array
    yPred = np.concatenate(yPred)
    yTestCom = np.concatenate(yTestArr)

    print("\nClassification Report:\n")
    print(classification_report(yTestCom, yPred))

def lr(XSetArr, ySetArr, xTestArr, yTestArr):
    lrModel = LogisticRegression()

    XTrainCom = np.concatenate(XSetArr, axis=0)
    yTrainCom = np.concatenate(ySetArr, axis=0)

    print("Fitting Logistic Regression Model...")
    lrModel.fit(XTrainCom, yTrainCom)

    print("Finished Fitting, Beginning Prediciton...")
    yPred = []
    for i in range(len(xTestArr)):
        preds = lrModel.predict(xTestArr[i])
        yPred.append(preds)
    
    # Flatten yPred if you want a single array
    yPred = np.concatenate(yPred)
    yTestCom = np.concatenate(yTestArr)

    print("\nClassification Report:\n")
    print(classification_report(yTestCom, yPred))

def nn(XSetArr, ySetArr):
    # Linearlly stack layers as a model
    nnModel = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(86,)),    # First layer uses RELU and 32 nodes
        tf.keras.layers.Dense(32, activation='relu'),                       # Next layer is the same
        tf.keras.layers.Dense(1, activation='sigmoid')                      # Last layer uses Signmoid function
    ])

    # Compile the Neural Network with the Adam optimizer using binary cross entropy as our loss and
    # 0.001 as the learning rate. We will also have another metric stored for us, accuracy
    print("Compiling Neural Network...")
    nnModel.compile(optimizer=tf.keras.optimizers.Adam(0.001), 
                    loss='binary_crossentropy',
                    metrics=['accuracy']
    )
    
    XTrainCom = np.concatenate(XSetArr, axis=0)
    yTrainCom = np.concatenate(ySetArr, axis=0)

    print("Finished Compiling, Fitting Neural Network Model...")
    history = nnModel.fit(
        XTrainCom, yTrainCom,
        epochs=10, batch_size=32,
        validation_split=0.2, verbose=0
    )

    print("Finished Fitting, Plotting Data...")
    plot_loss(history)
    plot_accuracy(history)

""" THE FOLLOWING TWO FUNCTIONS ARE FROM TENSORFLOW TUTORIALS """
def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Binary Cross Entropy')
    plt.legend()
    plt.grid(True)
    if not os.path.exists("Figures/Neural_Network"):
        os.makedirs("Figures/Neural_Network")
        plt.savefig("Figures/Neural_Network/Loss.png")
    plt.show()

def plot_accuracy(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    if not os.path.exists("Figures/Neural_Network"):
        os.makedirs("Figures/Neural_Network")
        plt.savefig("Figures/Neural_Network/Acc.png")
    plt.show()

def main():
    global dfList
    global labels
    askDownload = input("Do you need to download the datasets (y/n)?: ").lower().strip() == "y"
    if (askDownload):
        downloadData()

    mode = int(input("Enter dataset selection (0: Both, 1: CICIDS2017, 2: CSECICIDS2018): "))
    askGraph = input("Process subplot data (y/n)?: ").lower().strip() == "y"
    askPlot = input("Process any data (y/n)?: ").lower().strip() == "y"

    concatData(mode)

    #print(labels)

    #print(dfList[0].head())
    #print(labels)
    if (askPlot):
        columns = ["Total Fwd Packet", "Total Bwd packets", "Average Packet Size"]
        plotData(columns, "Time", "Number of Packets", askGraph, labels)

    colToDrop = ["id", "Flow ID", "Src IP", "Dst IP", "Timestamp"]

    # Prepare data for training, validation, and testing
    train = []
    valid = []
    test = []
    for df in dfList:
        tr, va, te = np.split(df.sample(frac=1), [int(0.6 * len(df)), int(0.8 * len(df))])

        train.append(tr)
        valid.append(va)
        test.append(te)

    trScale = []
    XTrain = []
    yTrain = []

    vaScale = []
    XValid = []
    yValid = []

    teScale = []
    XTest = []
    yTest = []
    count = 0
    # Scale values relative to mean
    for df in dfList:
        # OverSample allows us to balance the amount of data if we want
        trS, XTr, yTr = scaleDS(train[count], colToDrop, overSample=True)
        vaS, XV, yV = scaleDS(valid[count], colToDrop, overSample=False)
        teS, XTe, yTe = scaleDS(test[count], colToDrop, overSample=False)

        trScale.append(trS)
        XTrain.append(XTr)
        yTrain.append(yTr)

        vaScale.append(vaS)
        XValid.append(XV)
        yValid.append(yV)

        teScale.append(teS)
        XTest.append(XTe)
        yTest.append(yTe)

        count += 1

    '''
    # Check the shapes of the training data
    for i in range(len(XTrain)):
        print(f"XTrain[{i}] shape: {XTrain[i].shape}, yTrain[{i}] shape: {yTrain[i].shape}")
    '''

    # Naive Bayes
    #nb(XTrain, yTrain, XTest, yTest)

    # Logistic Regression
    #lr(XTrain, yTrain, XTest, yTest)

    # Neural Network Training
    nn(XTrain, yTrain)

    # K-Means Clustering with Principle Component Analysis


    # Clear large objects and force garbage collection
    del dfList
    gc.collect()  # Force garbage collection to free memory

if __name__ == "__main__":
    main()