# AutoPET24


## Brief Description - summary of experiments:

- Original data was used to train ResENCL model for three folds
- Data preprocessed: CT intensity clipping [-800:800] + copping by using TotalSeg; ResENCL model was used for training 3 folds
- Dice metrics of preprocessed were significantly surpassed the original model
- For the sake of prediction time; all the preprocssed data were used to train one single model (no CV, intentionally OF for training)
- To speed up the cropping, simple thresholding was tested so we don't need TotalSeg anymore!
- The threshold-based cropped data were used for prediction (fold0) to compare their results against the TotalSeg cropped based model (make sure cropping with Thr is correct).
- The resuls from previous step shows that XXXXX

## TODO:
1 - Add post processing to build the seg masks in the original size
2 - FP reduction?


 
