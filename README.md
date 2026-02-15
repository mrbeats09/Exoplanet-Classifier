# Exoplanet-Classifier
### What it is 
This CatBoost ML model takes in TESS Target Pixel Files and estimate on whether the detection was an exoplanet or not. 
Primarily looks for eclipsing binaries by taking a close look at centroid movement. 

### What it does 
- Takes in TESS Target Pixel Files containing flux data, centroid data, signal-to-noise ratios and other important information 
- Uses lightkurve to interpret and clean data, returning them as 32-bit single preciaion float values
- Uses a CatBoost (gradient boosting) model, which is trained on about 300 examples
- It can then be used on new data to predict whether the data suggests a valid exoplanet detection or not 
