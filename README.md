# CSGO-Round-Win-Predictor
# Introduction
The CS:GO Round Winner Prediction project. Within this repository, an extensive analysis and implementation of round outcome prediction in the popular first-person shooter game Counter-Strike: Global Offensive (CS:GO) are hosted. Machine learning techniques are harnessed to construct precise predictive models, thereby enriching comprehension of in-game dynamics.
# Setup
In this section, the groundwork for the project is established through the importation of essential Python libraries and tools. <br>
Among these are numpy and pandas for data manipulation, along with matplotlib and seaborn for data visualization. <br>
Additionally, requests are utilized for fetching the CS:GO dataset. <br>
# Data Collection
The commencement of our analysis involves the acquisition of the CS:GO dataset from OpenML via its API. <br>
The dataset, known as CSGO-Round-Winner-Classification, holds a wealth of gameplay data encompassing diverse features such as kills, deaths, bomb placements, and round outcomes. <br>
This dataset exploration enables ,to delve into the factors exerting influence on round-winning probabilities. <br>
# Data Preprocessing
Download and read the dataset. <br>
Extract and clean the data by removing unnecessary lines. <br>
Create a DataFrame with cleaned data. <br>
Encode the target variable 'round_winner' into numerical values (0 for CT side, 1 for T side). <br>
## Feature Selection
Not all features contribute uniformly to our predictive models. In order to ensure peak performance, feature selection is undertaken. <br>
This process involves evaluating the correlation between each feature and the target variable ('round_winner').<br>
By selecting the top features based on a designated correlation threshold, a refined dataset is constructed that retains solely the most influential attributes.<br>
# Data Visualization
Visualizing data insights is key to understanding underlying patterns. Data visualization techniques are used to uncover relationships between features and the target variable. <br>
Heatmaps provide a clear overview of feature correlations, while histograms showcase data distributions, shedding light on potential biases.<br>
# Data Normalization
Prior to model training, it's crucial to normalize the data to a consistent scale. <br>
This ensures that features with varying magnitudes don't unduly influence the model's performance.We achieve normalization by splitting the dataset into training and testing sets and applying the StandardScaler to scale feature values.

# K-Nearest Neighbors (KNN) Classifier
The K-Nearest Neighbors algorithm, a popular choice for classification tasks, is our first step toward building predictive models. <br>
Train a KNN classifier using the scaled training data and evaluate its accuracy on the test data.<br>
Additionally, hyperparameters are fine-tuned using GridSearchCV to extract optimal performance.<br>
# Random Forest Classifier
The journey continues with the implementation of a Random Forest classifier, known for its robustness and ability to handle complex relationships. <br>
Similar to the KNN approach, the classifier is trained, assess its accuracy, and fine-tune hyperparameters using RandomizedSearchCV.<br>

# Neural Networks
Neural Networks are computational models inspired by the organization of neurons in animal brains, known as biological neural networks. These networks are designed to simulate the behavior and functions of the brain, allowing machines to learn from data and perform various tasks. One common type of neural network is the feedforward neural network, which consists of layers of interconnected neurons.Feedforward Neural Network Architecture<br>
The neural network comprises multiple layers of neurons, each performing specific computations. The layers are organized in a sequential manner, where data flows through the network from the input layer to the output layer.
In this architecture:<br>
The input layer has 20 neurons, which corresponds to the shape of the input data.<br>
There are three hidden layers with 400, 200, and 200 neurons, respectively. The activation function used in these layers is the Rectified Linear Unit (ReLU), which introduces non-linearity to the model.<br>
The output layer consists of a single neuron with a sigmoid activation function, which is commonly used for binary classification problems.<br>
Train the neural network and the evaluate the model<br>
