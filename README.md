# Exoplanet-Classifier

### Introduction 
This CatBoostClassifier ML model classifies between exoplanet detections and false positives (such as Eclipsing Binaries) via the use of flux data and centroid movement information extracted from TESS Target Pixel Files of a 2-minute cadence. 
The examples used to train this model were identified from the Tess Objects of Interest (TOI) database along with the corresponding labels, and these examples were matched with their corresponding Target Pixel Files to get the necessary information. 

### Training the model - what the scripts do 
- Queries the Caltech TOI database for 1000 examples - 500 Confirmed Positives and Known Positives (CP, KP) as well as 500 False Positives (FP) - at random and fetches the values 'tic_id', 'tfowpg_disposition' and 'sectors', storing them in a .csv file
- Uses 'tic_id' to gather the corresponding TESS Target Pixel Files, extracting 1000 timestamps of flux intensity values, centroid row positions, and centroid column positions
- Stores this information in a seperate .csv file along with the corresponding label (CP, KP or FP)
- Fetches this information from the .csv file and places all data into a Pandas DataFrame
- Calculates statistical parameters from the 1000 timestamps for each property of each example (such as mean, standard deviation, min, max, ptp, IQR, etc.)
- Utilises K-Folds as a means of Cross-Validation in order to evaluate the model on the entire dataset (to better estimate the model's accuracy)
- Trains the model using the CatBoostClassifier (a reputed Gradient Boosting algorithm) 
- Outputs a confusion matrix showcasing the accuracy of the model
- Saves the model itself in a .cbm file for later use, along with a few other key files

> NOTE: Since not all TOI examples necessarily have complementary Target Pixel Files, less than 1000 examples might be saved. 
> For our testing purposes, only 900 examples were utilised for this reason. 
