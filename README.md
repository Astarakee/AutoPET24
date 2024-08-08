# AutoPET24


## Brief Description - Summary of experiments:

- Original data was used to train ResENCL model for three folds
- Data preprocessed: CT intensity clipping [-800:800] + copping by using TotalSeg; ResENCL model was used for training 3 folds
- Dice metrics of preprocessed significantly surpassed the original model
- For the sake of prediction time; all the preprocessed data were used to train one single model (no CV, intentionally OF for training)
- To speed up the cropping, simple thresholding was tested so we don't need TotalSeg anymore!
- The threshold-based cropped data were used for prediction (fold0) to compare their results against the TotalSeg cropped-based model (make sure cropping with Thr is correct).
- The comparison resulted in similar results when using MetricsReloaded. Conclusion: Using threshold-based cropping results in similar Dice as TotalSeg in the inference phase.
- UMamba was used for training the whole dataset (all_fold)
- MedNeXt failed to converge!
- postprocessing: TotalSeg masks were analyzed; might be helpful, but 5 mins inference time!
- Split FDG and PSMA: ResENC model, slightly better performance; depending on classification power.
