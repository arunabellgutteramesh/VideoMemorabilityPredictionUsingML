# VideoMemorabilityPredictionUsingML
Some videos have more impact than the others resulting in higher memorability scores for such videos.  Using various ML algorithms, such memorability scores are predicted. 

# Assignment Problem Statement

Predict short-term and long-term memorability scores using either the video features like C3D, HMP or semantic feature like the captions (explaining the video content).

# Solution 

After varied explorations, it was observed that, Semantic feature – Captions gave better results compared to video features. Hence, extensive work was done on semantic feature. The captions were cleaned by removing special characters, converting captions into small case and removing stop words. The cleaned words were used to create a bag of words. This bag of words was run with TfIdfVectorizer to obtain features. These features were sent as independent variables to my Machine Learning (ML) model. TfIdf is a statistical measure used to evaluate how important a word is to a document in a collection or corpus. So, the TfIdfVectorizer calculates the TfIDf value of each word in the given corpus and forms a feature. 

Similarly, the bag of words was also run with CountVectorizer. But, TfIdfVectorizer outperformed CountVectorizer. CountVectorizer calculates the frequency of occurrence of a word in the given corpus. Video and Semantic features – here, C3D and captions - were combined and sent as independent variables to ML model. Captions were run through with TfIdfVectorizer before sending to the ML model. C3D feature was taken as is. Even though this performed better than C3D feature alone, it failed to perform better than captions alone.

It has been said that few terms/visuals have more positive impact on memorability than others. These terms are also given in their paper with the coefficiency of their effect. Contrary to the popular belief, terms pertaining to nature had a negative effect i.e., less memorability score and terms pertaining to people or indoor actions had a positive effect i.e., more memorability score. Using this concept, I gave certain extra weights to the terms with positive coefficiency (terms were given in the winning paper). These terms were searched in captions, if found the weight for the caption was cumulatively increased. This model of mine performed the best with Random Forest Regression Model and n_estimators=100.

In my exploration I learnt that the model with weighted captions worked best. Hence, I used the same model for my final computation. Therefore, my ML model is on semantic feature (captions), with TfIdfVectorized features, along with weighted captions.

# Report

A detailed report of this assignment is available in the repo.
