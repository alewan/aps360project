# APS360 Final Report
### LEC02-18
### Submitted: April 9th, 2020
### Total Word Count: 2489 / 2500

# Introduction

The motivation behind this project is to create a way to recognize and categorize facial expressions such as anger, and sadness from images, which is a difficult subject in machine learning. The goal is to determine emotions from facial expressions through images captured in a set of video clips. This will be useful for companies aiming to create apps for translating American Sign Language (ASL). It would assist in identifying the message that one attempts to convey more accurately as both the hand gesture and emotion recognition is equally important. Machine learning is a reasonable approach for this project because there is a pattern that can be observed in vast amounts of facial expression data.

# Illustration / Figure

![Illustration](/imgs/Illustration.png)

**Figure 1:** System Diagram with Major Components of the Project

# Background & Related Work

The field of facial recognition, and the subset of analyzing facial expressions, is rapidly growing, with many contributions from large tech companies including IBM [1], Microsoft [2], and Amazon [3]. Microsoft’s Azure provides a service where an image can be uploaded, and a face will be detected along with additional attributes such as emotion [2].  In 2016, researchers at Microsoft also created a model to detect human emotions from video and audio data [3]. Amazon’s Rekognition functions similar to Azure, where an image can be uploaded to a web service, and emotional categories for the given face will be received [4]. Our project aims to improve the accuracy of Rekognition’s general solution with our own dataset.

# Data Processing

The sources of our data stem from the RAVDESS dataset containing 24 actors acting short clips of emotion [4]. After the data files have  been downloaded, they are named xx-xx-xx-xx-xx-xx-xx.mp4, the relevant xx are the first and the third, which represent the modality (Video only, audio only etc) and the emotion respectively. To process the data, we first create 3 directories of train/val/test each with 5 subdirectories of the emotions (e.g. neutral/calm, sad, etc). We then separate the videos from each actor into the groups 1-14, 15-19, 20-24 (for actors 1-24) for train/val/test sets respectively to ensure no actor is in both the train and validation or train and test set or test and val set.

The next step is to use ffmpeg on every mp4 file within that actor’s directory that has the first xx as 01 for video only. Using the metadata, we know the label for the data is in the third number of the file name, through this we can sort the output into the correct folders such as 01 for neutral, 03 for happy, etc.

This process was done with a python program that would call ffmpeg on the video, create a globally unique name for each jpg, and then move the jpg files into the correct directory based on the mp4 file name. This process is repeated for every video file in the actor’s directory, and every actor in the RAVDESS dataset. When the data is loaded using the imagefolder method, a transform using torchvision.transforms.compose can be used to downsample (and centre crop) the images to a size of 360x640 (Height, width).

Train: 1410 Angry/Fearful, 1049 Neutral/Calm, 744 Disgust, 687 Happy, 687 Sad

Validation: 497 Angry/Fearful , 364 Neutral/Calm, 273 Disgust,  243 Happy, 248 Sad

Test: 471 Angry/Fearful, 361 Neutral/Calm, 243 Disgust, 230 Happy, 241 Sad

There are two reasons for the differing sizes of the classes. The first reason is that the mp4 files have differing length with the same frame rate, since we sample on frames per second, longer videos will have more samples.The second reason is that the original dataset had a video for angry and a video for fearful but we combined those classes as well as the classes for neutral/calm. Since even state of the art systems have difficulty differentiating between the two the classes are combined for better model performance.

The following are example pictures from after the data processing (Each of a different emotion with a different actor):

![Correct Sample Data](/imgs/correct_sample.png)

**Figure 2:** From top left to bottom right: Angry, calm, disgust, fear, happy, neutral, sad, surprised.

# Architecture

There are two models that the team has experimented with. The first a model with both CNN and LightGBM to increase the final result accuracy while another is only the result of AlexNet.

The following shows the basic structure used for the CNNand LightGBM model. First, CNN was used to extract the features from the input. Once tuned with the hyperparameters, the epoch number in which resulted in the highest accuracy was given to the LightGBM in order to increase the final accuracy.

![Model LightGBM](/imgs/model_lightgbm.png)

**Figure 3:** CNN with LightGBM model

The following diagram shows the basic structure and specific parameters used for the best AlexNet model, the final model. As seen in the diagram, the AlexNet model has 3 fully connected layers at the end with ReLU activation function after the features extraction. The hyperparameters used were: the learning rate was 0.001, batch size of 32 with 85 epochs.

AlexNet is chosen because we have a classification problem, with a fixed input size and want to take advantage of the spatial locality of a person's face when expressing emotions. The AlexNet can also detect the same features across entire images. The ANN layers after the AlexNet features extraction are for evaluating the features that the AlexNet extracts for classification. Softmax is used at the end to bucket the ANN’s output into the 5 classes for the emotions.

![Model](/imgs/model.png)

**Figure 4:** Best AlexNet Model

# Baseline Model

The baseline model that we are using is Amazon Web Services (AWS) Rekognition [5], pictured in the diagram below (Figure 5), a machine learning-based service provided by cloud services provider AWS. Rekognition allows users to send in images and receive percentage emotional categorizations (Figure 6, right-hand side) through either a web interface (Figure 6, left-hand side) or Amazon’s CLI/libraries. We intend to use the top prediction (i.e. highest percentage) produced by AWS as the baseline. Running on our test set, Rekognition achieves 48% top prediction accuracy.

![AWS Rekognition Outline](/imgs/baseline1.png)

**Figure 5:** AWS Rekognition Diagram

![AWS Rekognition Baseline](/imgs/baseline_aws.png)

**Figure 6:** AWS Rekognition Web-based Interface & Emotional Categorizations Returned by AWS Rekognition

# Quantitative Results

The following graphs show the results of the training and validation accuracy and loss on the training data. We used 85 epochs and the best validation and test accuracy was given through the epoch with the lowest validation loss. As seen in the graph, the model is able to identify the emotion from the image extracted 50% of the time.

![Accuracy Graph](/imgs/accuracy_graph.png)

**Figure 7:** Training and Validation Accuracy and Loss

The final accuracy of the AlexNet model came out to be the following: the training accuracy was 84.34%, the validation accuracy of 55.76% and the test accuracy of 50.63%.

The following is the confusion matrix resulted from the test data set. As seen in the matrix, 4 out of 5 labels have mostly been correctly labelled by the model. The only label that the model had the hardest time was the Sad label as it mostly got labelled as AngryFearful. However, other than the Sad label, the model was able to correctly label most of the data set given.

**Table 1:** Confusion matrix of the validation data set.

| | AngryFearful | Disgust | Happy | NeutralCalm | Sad | Total |
|-|--------------|---------|-------|-------------|-----|-------|
| AngryFearful | 280 | 64 | 37 | 78 | 38 | 497 |
| Disgust | 86 | 290 | 24 | 38 | 108 | 546 |
| Happy | 16 | 10 | 386 | 72 | 2 | 486 |
| NeutralCalm | 48 | 4 | 67 | 345 | 13 | 477 |
| Sad | 140 | 210 | 26 | 26 | 94 | 496 |
| Total | 570 | 578 | 540 | 559 | 255 | 2502 |

# Qualitative Results

The following table shows the results of the labelled results of the model. The images of the left column the model was able to correctly predict the images to the 5 classes respectfully. When the image (data extracted) has clear facial expression of the actors such as the positions of the mouth, eyebrows etc., (a clear indication of the facial motion has changed) then the model is able to correctly predict it. However, as seen in the right hand side, some facial structures the model is unable to differentiate between (e.g. sadness and disgust), so there is a 50% chance it will be the correct label. Another factor is that once the model determines the expression to be one label from the random sampling and the face shows barely any change in expression, the model will keep on predicting the same label.

**Table 2:** Results of hand picked examples from the validation dataset.
|Label |Image |Prediction |Image |Prediction |
|------|------|-----------|------|-----------|
| Disgust | ![Data Sample](/imgs/q1.png) | Disgust | ![Data Sample](/imgs/q2.png) | Sad |
| AngryFearful | ![Data Sample](/imgs/q3.png) | AngryFearful | ![Data Sample](/imgs/q4.png) | NeutralCalm |
| NeutralCalm | ![Data Sample](/imgs/q5.png) | NeutralCalm | ![Data Sample](/imgs/q6.png) | AngryFearful |
| Sad | ![Data Sample](/imgs/q7.png) | Sad | ![Data Sample](/imgs/q8.png) | Disgust |
| Happy | ![Data Sample](/imgs/q9.png) | Happy | ![Data Sample](/imgs/q10.png) | NeutralCalm

# Evaluate model on new data

In data preprocessing, the overall data was split into 60% train, 20% validation, and 20% test. The data was also split to ensure that actors do not appear across different datasets. This left 5 actors, seen in Figure 9, that went completely unseen by both the model and developers until the final model was determined and tested at the very end.

![Test Set Actors](/imgs/test_set_actors.png)

**Figure 9:** Actors used in test-set.

Our final model has a validation accuracy of 55%. Our baseline model, which is a state of the art tool, has an accuracy of 48% on our test set. Our model achieved a 50% accuracy on the test set. This is a drop from the validation accuracy, which is expected given that the model has never used these faces in training and tuning.  However, our model still outperforms our baseline model, which in our case, is a sophisticated, state of the art tool.

The confusion matrix for our test set, seen in Table 3 below, shows an increased number of errors, but these errors are consistent with those seen in the validation set, Table 1. For example, the model confused a significant portion of the Sad label with AngryFearful in both validation and test sets. However, the model misclassified significantly more with NeutralCalm in the test set than it did in the validation set. This appears to be due to there being  more mislabeled images in the test set compared to the validation set. Actors with NeutralCalm facial expressions, as seen in Figure 10, were labeled as a NonNeutralCalm emotion.

**Table 3:** Confusion Matrix of the model on the test set.

| | AngryFearful | Disgust | Happy | NeutralCalm | Sad | Total |
|-|--------------|---------|-------|-------------|-----|-------|
| AngryFearful | 277 | 35 | 13 | 121 | 25 | 471 |
| Disgust | 86 | 304 | 30 | 52 | 14 | 486 |
| Happy | 78 | 10 | 240 | 128 | 4 | 460 |
| NeutralCalm | 82 | 23 | 26 | 326 | 20 | 477 |
| Sad | 168 | 74 | 2 | 182 | 56 | 482 |
| Total | 691 | 446 | 311 | 809 | 119 | 2376 |

![Misslabeled Actors](/imgs/misslabeled_actors.png)

**Figure 10:** Test images predicted as NeutralCalm but mislabeled.

Finally, the team also tested the model on images of ourselves. These images varied in that the clothes worn were not black, and the background was not perfectly white. The model’s performance remained at 50%. Overall, the model performed slightly above our baseline, on new unseen test data. This exceeds our expectation, since it passes the accuracy of AWS’ Rekognition.

# Discussion

Overall, we believe that our model is performing well, given the unique challenges and difficult nature of the problem. While we will discuss the difficulty of the project in greater detail later, we obtained accuracy exceeding top cloud services providers with significantly more resources which is a good achievement.

The qualitative results give us a strong starting point for interpreting the results. As we can see in the images from Table 3, there are some difficulties with random sampling images from a video. There are natural breaks in speech, and people may not always be clearly expressing the emotion that is in the correct label of a video, which can impair the ability of our model to perform better since even a person may struggle with this type of classification. There is an obvious solution to this, more effort in processing the data to try to pull better samples, but this is time-consuming and potentially quite labour intensive.

An interesting peculiarity of the results was that increasing the model complexity did not help much. When using LightGBM in an attempt to improve the model performance or learn patterns of misclassification, the LightGBM tree was not able to significantly improve performance (hence why it was not included in the final model or results). This happened despite the patterns in the model predictions evident in the quantitative performance.

The quantitative results also provide an interesting lens through which to view model performance. From the confusion matrix above, it is possible to notice that the network was having trouble with a couple of categories in particular. Firstly, it was only able to correctly predict less than 20% of the sad images, classifying them as disgust 42% of the time and angry/fearful 28% of the time. Furthermore, when attempting to classify images labelled as disgust, though the model made the correct prediction most of the time, sad and angry/fearful were the two categories that the model predicted most frequently when it was wrong. This hints that there may be an underlying relationship between these emotions that make them harder to distinguish for the model than the other emotions.

In summary, there are three key lessons that we learned through the results. Firstly, emotion identification without greater context is a very difficult problem. Adding the challenge of random sampling further complicates this classification problem. However, there are some bright spots as well. AlexNet performed well in classifying features, and was able to significantly out-perform alternatives that we tried. Finally, it appears that certain emotions are harder to separate out than others as is elucidated above.

# Ethical Considerations

The RAVDESS dataset [5] provides us information with unrestricted access. The 24 actors within the dataset range in age from 21-33 years. The ethnic diversity of the actors includes: 20 actors self-identifying as Caucasian, and 2 as East-Asian, and 2 as other [5]. The ethnic diversity of the dataset will limit our model to the age and ethnic range of the actors used for training. Another ethical issue related to the models use; steps will need to be taken to ensure proper consent from a person before their face is analyzed. To address this, we ensured that the racial diversity of our actors was equally distributed over our train, validation and test sets.

# Project Difficulty / Quality

Determining emotions from still images of facial expressions is a difficult task. Tech giants with significant resources such as Amazon and Microsoft had a difficult time achieving better than ~35% accuracy on an 8 class problem and ~48% on a 5 class problem. Our model achieved a test accuracy of 50.63%, which exceeded expectations of state of the art technology. This is in part due to AWS and Azure having to generalize to the world’s data rather than our model which always had a centred face on a white background.

The difficulty of this problem comes in two forms, the first being the quality of the labels. As seen in figure 12 below, the methodology used to extract images to train off of leads to less than ideal image-label pairs. Better extraction methods would rely on knowing when to sample a video on “expressive” emotions, which in itself would require a large human investment or its own model to train.

![Misslabeled Actors](/imgs/misslabeled_actors_2.png)

**Figure 12:** Showing the labels of the images not being representative of the images.

The second part of the difficulty comes from the continuity of human interactions. In Figure 12 above, the still images are not representative of a human readable emotion. Instead humans can understand emotion through context.. Context can be added in the form of audio cues or text cues; however, the purpose of this model was visual cue-based only. To add context, a series of images could be used with an RNN or LSTM model in order to better understand continuity but that adds a new level of complexity. How frequently do you have to sample in order for the images to maintain continuity, how many samples would be fed into the model, and how fast would the model train if all those samples were used? Due to the structure of RNNs the training time increase would not scale linearly with images. As such with the computational resources accessible, it would not be feasible to train a model using many past images as context.

In conclusion, the image labels and the continuity of human emotion interactions make the problem of determining a human’s emotion from a still image significantly complex.

# Github Link

[Project Link](https://github.com/alewan/aps360project)

# References

[1] Cloud.ibm.com. (2020). IBM Cloud Docs. [online] Available at: https://cloud.ibm.com/docs/services/tone-analyzer?topic=tone-analyzer-about#about [Accessed 22 Feb. 2020].

[2] Docs.microsoft.com. (2020). Face Documentation - Quickstarts, Tutorials, API Reference - Azure Cognitive Services. [online] Available at: https://docs.microsoft.com/en-us/azure/cognitive-services/face/ [Accessed 22 Feb. 2020].

[3] Bargal, S., Barsoum, E., Ferrer, C. and Zhang, C. (2016). Emotion recognition in the wild from videos using images. Proceedings of the 18th ACM International Conference on Multimodal Interaction - ICMI 2016. [online] Available at: https://dl.acm.org/doi/abs/10.1145/2993148.2997627 [Accessed 22 Feb. 2020].

[4] S. Livingstone, F. Russo. The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English, University of Ryerson, May 16, 2019, Accessed On: February 2, 2021. Available: https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0196391&fbclid=IwAR0pMF9vaxEvCucqjm2DJ1TH6CUv7JpBD79vi8qJcCAdHzJjJ4X2pFGDv_E

[5] "Amazon Rekognition – Video and Image - AWS", Amazon Web Services, Inc., 2020. [Online]. Available: https://aws.amazon.com/rekognition/. [Accessed: 22- Feb- 2020].

