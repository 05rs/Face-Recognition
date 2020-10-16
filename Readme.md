##  End to End Face-Recogination in `progress`...


###  Status

- Image Extraction and `Preprocessing Pipeline` complete.
  - Directory based retrival
  - Face Extraction using `DLib` and Face Alignment (eyes nd mouth landmark based alignment) 

- `Model Pipeline` complete in Tensorflow.
  - Converted orignal model in csv to binary for tensorflow.
  - Creating `Siamese Net` adding a triplet loss layer.
  - Used Tensorflow `Semihardloss`.

- `Hard Triplet Generator` completed for further Fine-Tuning.
  - Made fast custom Image Data Generator for implementing Semi-Hard online 
    Triplet Generator.
    
### Features
- Online hard triplet generator.

### Next Todo:

- Visualization
- Model evaluation and metrics

