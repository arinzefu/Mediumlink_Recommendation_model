# Mediumlink_Recommendation_model
Torch model with text embeddings

The dataset is from https://www.kaggle.com/datasets/viniciuslambert/medium-data-science-articles-dataset


The aim of this Python script is to build a recommendation model that suggests URLs based on given input texts. The script begins by preprocessing the dataset, then consolidates text-related columns ('title,' 'subtitle,' 'tag,' 'author') into a new 'text' column. It normalizes numeric features ('claps' and 'responses') using StandardScaler.

For text embedding, the script utilizes the BERT transformer. This involves tokenizing, extracting embeddings, and combining them with numeric features. A straightforward neural network with a linear layer is created for the recommendation model.

The custom RecommendationModel class employs Mean Squared Error as the loss function and the AdamW optimizer. The training loop is designed to optimize the recommendation model's parameters based on the specified loss function and training data. The objective is for the model to learn to assign higher scores to positive samples (actual recommendations) compared to randomly chosen negative samples. This loop iterates for the specified number of epochs, gradually enhancing the model's ability to provide accurate recommendations.

Once the training is complete and the model is saved, the script proceeds to evaluate the model on a custom text input. However, it encounters some code line errors while attempting to obtain the recommended URL. The #evaluation mode needs to be written to retrieve a URL
