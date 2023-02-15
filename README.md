# Multi lingual Video Emotion Detection
This project focuses on classifying the emotion of videos based on facial and audio features.
I used separate EfficientNet neural network architecure to classify audio and facial expressions.
The models were trained on kaggle datasets, both for audio and images.
Image training was done in batches using a custome dataloader.
Audio training was a bit tricky because the model had to be independent of any language. To achieve that I had to convert audio into 2d representation that showed frequency
and amplitude. The model was then trained to pick up these features.
The trained models were then saved in a PT file.

Initially, faces of those in the videos were extracted and then sent to the trained model. At the same time audio was also extracted and was recieved by its own trained model.
The output was based on time interval. After a few seconds, the results showed the emotions of the past video segment on the basis of audio and facial images.
