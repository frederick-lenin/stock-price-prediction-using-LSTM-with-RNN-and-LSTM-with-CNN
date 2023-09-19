# stock price prediction using LSTM-RNN and LSTM-CNN

# ABSTRACT:
In the current financial sector, predicting stock prices is a key activity, and most of the prediction algorithms are typically based on machine learning and deep learning algorithms. Nonstationary time series data are challenging for these algorithms to handle, though. For predicting stock price, deep learning methods including convolutional neural networks (CNNs), recurrent neural networks (RNNs), and long short-term memory (LSTM) networks have been extensively used. Because they can store historical data, LSTMs are extremely effective prediction methods. We suggested a fresh deep learning prediction technique. This study suggests a technique for forecasting changes in stock prices using recently updated stock prices. We evaluate variations in accuracy while concatenating two alternative top algorithms, LSTM-RNN and LSTM-CNN.
Keywords – Stock price; LSTM; RNN; CNN; deep learning

# INTRODUCTION
# 1.1 MOTIVATION OF WORK
The motivation behind using LSTM (Long Short-Term Memory) and RNN (Recurrent Neural Network) models for stock price prediction, as well as combining LSTM with CNN (Convolutional Neural Network), is to 
leverage the power of deep learning to capture and learn from the sequential and temporal patterns present in stock market data. Here are a few reasons why these models are commonly used for stock price 
prediction: 
1. Sequential data analysis: Stock market data is inherently sequential, where the order of data points matters. LSTM and RNN models are designed to effectively model and capture dependencies between past and present data points, making them suitable for analysing and predicting stock price movements.
2. Long-term dependencies: LSTM, a variant of RNN, is specifically designed to address the vanishing gradient problem associated with traditional RNNs. It can retain information for longer periods, allowing it to capture longterm dependencies in the data. This is crucial for stock price prediction, as past prices and trends can have a significant impact on future prices.
3. Non-linear patterns: Stock market data exhibits complex and nonlinear patterns, which cannot be easily captured by traditional statistical models. Deep learning models, such as LSTM and CNN, have the ability to learn complex patterns and relationships from the data, enabling them to make more accurate predictions.
4. Feature extraction: CNNs are widely used in computer vision tasks for their ability to automatically extract relevant features from raw data. Similarly, in stock market prediction, CNNs can be used to extract relevant features from historical price data, allowing the LSTM model to focus on capturing temporal dependencies.
5. Handling high-dimensional data: Stock market data often consists of multiple variables or features, such as price, volume, and technical indicators. LSTM and CNN models can handle high-dimensional data efficiently, allowing them to capture and learn from the multi-dimensional relationships in the data.
By combining LSTM and RNN models with CNNs, we can leverage the strengths of both architectures. The CNN can extract meaningful features from the input data, and the LSTM or RNN can effectively capture the temporal dependencies within those features. This combined approach has shown promising results in capturing complex patterns and improving the accuracy of stock price predictions. Overall, the motivation behind using LSTM, RNN, and LSTM-CNN models for stock price prediction is to leverage their ability to capture sequential patterns, handle high-dimensional data, learn complex relationships, and make accurate predictions based on historical price data.
# 1.2 PROBLEM STATEMENT
The objective of this project is to develop a predictive model for stock price movements based on historical stock market data. The problem can be defined as follows:
1. Data Collection: Historical stock market data needs to be collected for the target stock(s) from reliable sources. This data should cover a significant time period to capture various market conditions and trends.
2. Feature Selection: Relevant features, such as price-related attributes (opening, closing, high, low), trading volume, and technical indicators, need to be identified and extracted from the collected data. The selection of features should be based on their importance and relevance to stock price movements.
3. Data Pre-processing: The collected data needs to be pre-processed to handle missing values, outliers, and any inconsistencies. Additionally, normalization or scaling techniques may be applied to ensure that the features are on a comparable scale and prevent bias in the model.
4. Time Series Analysis: Stock market data is inherently sequential, with the order of observations playing a crucial role. The model should be designed to capture the temporal dependencies and patterns present in the data.
5. Model Training: Various deep learning algorithms can be explored for stock price prediction, such as LSTM (Long Short-Term Memory), RNN (Recurrent Neural Network), or other regression models. The selected model should be trained on the pre-processed data, optimizing its parameters to minimize the prediction error.
6. Model Evaluation: The trained model needs to be evaluated using suitable evaluation metrics, such as mean squared error (MSE), mean absolute error (MAE), or accuracy. The evaluation should be performed on a separate testing dataset that was not used during the training phase to assess the model's generalization ability.
7. Predictive Performance: The effectiveness of the model will be assessed based on its ability to accurately predict the future price movements of the target stock(s). The model should demonstrate a reasonable level of accuracy and reliability in capturing the market trends.
By addressing the problem statement, the aim is to develop a robust and accurate predictive model that can assist investors, traders, and financial professionals in making informed decisions related to stock market investments. The model should provide valuable insights into the potential direction of stock prices, enabling users to optimize their trading strategies and mitigate investment risks.

# LITERATURE SURVEY
# 2.1 GENERAL
Stock price prediction is a challenging task in the financial industry, and the application of deep learning techniques such as recurrent neural networks (RNNs) and long short-term memory (LSTM) models has gained significant attention. In this literature survey, we explore ten noble studies that have utilized LSTM and RNN models for stock price prediction, highlighting their methodologies, datasets, and performance metrices.
# 2.2 RELATED WORK
Jooweon Choi et al.[1], provided the multilayer perceptron-based model, the hybrid information mixing module, is applied to the stock price movement prediction to conduct a price fluctuation prediction experiment in a stock market with high volatility. In addition, the accuracy, Matthews correlation coefficient (MCC) and F1 score for the stock price movement prediction were used to verify the performance of the hybrid information mixing module.
Disadvantages
Increased Complexity: Implementing the HIMM technique adds complexity to the prediction model. It involves integrating different types of data sources, such as historical price data, news articles, social media sentiment, and other relevant information. Managing and processing diverse data types and sources can increase the complexity of the prediction model, making it more challenging to develop and maintain.
Data Integration Challenges: Incorporating various data sources into the HIMM framework requires careful data integration. Different data sources may have varying formats, structures, and levels of quality. Ensuring proper alignment, preprocessing, and integration of these heterogeneous data can be time-consuming and prone to errors. Additionally, managing the timeliness and reliability of real-time data sources can pose additional challenges.
Nagaraj Naik et al. [2], examined the stock crisis prediction based on Extreme Gradient Boosting (XGBoost) and Deep Neural Network (DNN) regression method. The performance of the model is evaluated based on Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Square Error(RMSE). HFS based XGBoost method was performed better than HFS based DNN method for predicting the stock crisis. The experiments considered the Indian datasets to carry out the task.
Disadvantages
XGBoost has several hyperparameters that need to be tuned to achieve optimal performance. This can be time-consuming and requires domain knowledge or extensive experimentation. Selecting the right combination of hyperparameters can be challenging, and improper tuning may lead to overfitting or underfitting.
XGBoost can be computationally expensive, especially when dealing with large datasets or complex models. Training an XGBoost model can require substantial memory and processing power. In some cases, this can limit its applicability on resource-constrained systems.
Saud S. Alotaibi [3], proposed the final predicted results that are acquired from the Optimized Neural Network (NN). To make the precise prediction, the training of NN is carried out by the proposed RDAWA via finetuning the optimal weight. Finally, the performance of the proposed work is compared over other conventional models with respect to certain measures.
Disadvantages
Limited Generalization: While the ensemble technique aims to improve generalization by combining multiple models, the use of a novel hybrid Red Deer-Grey algorithm may introduce limitations in generalizing to other datasets or market conditions. The algorithm's performance and effectiveness may vary across different datasets or time periods, and it may not perform as well in scenarios that differ significantly from the training data.
Complexity and Computational Resources: Implementing the hybrid Red Deer-Grey algorithm can be computationally intensive and require significant computational resources. The algorithm combines two different optimization techniques, which can increase the complexity of the overall model. Training and evaluating multiple models within the ensemble, especially when using a large number of features, may demand substantial processing power and memory resources.
Sondo Kim et al. [4], examined and suggested utilizing the adjusted accuracy derived from the risk-adjusted return in finance as a prediction performance measure. Lastly, we confirm that the multilayer perceptron and long short-term memory network are more suitable for stock price prediction. This study is the first attempt to predict the stock price direction using ETE, which can be conveniently applied to the practical field.
Disadvantages
Dependency on Feature Engineering: The success of the model relies heavily on feature engineering, particularly in identifying relevant and informative features. The selection and engineering of features require domain expertise and understanding of the factors influencing stock price movements. The quality and relevance of the engineered features significantly impact the model's performance, and improper feature engineering may lead to suboptimal predictions.
Model Complexity and Overfitting: Machine learning techniques can introduce model complexity, increasing the risk of overfitting. Overfitting occurs when the model becomes too specific to the training data, leading to poor generalization on unseen data. Complex models and a large number of features can exacerbate this risk. Careful regularization, cross-validation, and monitoring of the model's performance on out-of-sample data are necessary to mitigate overfitting.
Yaohu Lin et al. [5], examined the measures such as big data, feature standardization, and elimination of abnormal data can effectively solve data noise. An investment strategy based on our forecasting framework excels in both individual stock and portfolio performance theoretically. However, transaction costs have a significant impact on investment. Additional technical indicators can improve the forecast accuracy to varying degrees. Technical indicators, especially momentum indicators, can improve forecasting accuracy in most cases
Disadvantage
Model Complexity and Maintenance: Ensemble machine learning techniques can introduce additional complexity to the model. Combining multiple models and integrating feature engineering schemes can make the overall model more complex and difficult to maintain. The complexity may increase the time and effort required for model development, testing, and deployment. Additionally, as new data becomes available or market conditions change, updating and adapting the model can be challenging.
False Signals and Market Volatility: Despite the potential effectiveness of candlestick charting and ensemble machine learning techniques, stock market trends are influenced by various factors, including market volatility and unexpected events. False signals and incorrect predictions can occur due to sudden market shifts, news events, or external factors that cannot be captured solely through historical price data and candlestick patterns. The model may struggle to handle extreme market conditions or periods of high volatility.

# III. PROPOSED SYSTEM
LSTM
LSTM (Long Short-Term Memory) is a type of 
recurrent neural network (RNN) architecture that is 
designed to effectively capture long-term dependencies 
and handle the vanishing gradient problem, which is 
common in traditional RNNs. LSTM achieves this by 
introducing memory cells and gating mechanisms that 
control the flow of information.
Here is an overview of how LSTM works:
1. Memory Cells: LSTM introduces memory cells as the 
core component. These cells can maintain information 
over long sequences and determine when to remember 
or forget information.
Each memory cell has an internal state, which is updated 
and passed along the sequence to retain relevant 
information.
2. Gates: LSTM utilizes gating mechanisms to regulate 
the flow of information in and out of the memory cells.
There are three types of gates in an LSTM unit:
Forget Gate: Determines which information to 
discard from the previous cell state.
ft = σ (Wf* [ht-1, xt] + bf)
ft = Forget gate for current timestamp.
σ = Sigmoid layer returns 0(forget everything) 
to 1(keep as it is).
Wf = Weight matrix
ht-1 = Hidden layer of previous timestamp 
xt = Input of current timestamp.
bf = Bias vector for forget gate.
Input Gate: Decides which new information to 
update in the current cell state.
it = σ (Wi*[ht-1, xt] + bi) 
Ct’ = tanh(Wc [ht-1, xt] + bc)
tanh = tanh layer returns -1 to 1
bc = bias vector for input gate.
Output Gate: Controls the information to be 
outputted from the current cell state.
ot = σ (Wo*[ht-1, xt] + bo)
ht = ot* tanh (Ct)
bo = bias vector for output gate.
Ct = cell state of current timestamp.
3. Computation Steps: At each time step, an LSTM unit 
takes three inputs: the current input, the previous hidden 
state, and the previous cell state.
The LSTM unit performs the following computations:
Compute the forget gate value, which determines which 
information to forget from the previous cell state.
Compute the input gate value, which determines the 
new information to update in the current cell state.
Compute the candidate cell state, which is a 
combination of the current input and the input gate 
value. Update the current cell state by combining the 
forget gate value with the previous cell state and adding 
the candidate cell state. Compute the output gate value, 
which determines the information to output from the 
current cell state. Update the hidden state by applying 
the output gate value to the current cell state.
4. Training:
During the training process, LSTM models are 
optimized to minimize a specific loss function, such as 
mean squared error (MSE), by adjusting the parameters 
of the network. This is typically done using 
backpropagation through time, where the gradients of 
the loss function are calculated and used to update the 
weights and biases of the LSTM units.
5. Prediction:
After training, the LSTM model can be used to make 
predictions on new, unseen data.
The model takes in the input sequence, and the 
computations described above are performed iteratively 
to generate predictions for each time step.
LSTM's ability to maintain and update the memory 
cells, along with the gating mechanisms, allows it to 
capture long-term dependencies and handle information 
flow effectively over time. This makes LSTM wellsuited for tasks involving sequential data, such as stock 
price prediction, natural language processing, and time 
series analysis.
RNN
RNN (Recurrent Neural Network) is a type of neural 
network designed to process sequential data by 
capturing and utilizing information from previous steps. 
It has a recurrent structure that allows it to maintain a 
hidden state that carries information from previous steps 
and pass it along to the next steps.
Here is an overview of how RNN works:
1. Sequential Data Processing: RNN is designed to 
process sequential data, such as time series or text data, 
where the order of the input elements matters.
Each element in the sequence is processed one at a time, 
and the hidden state is updated and passed along to the 
next step.
2. Hidden State: The hidden state is a vector that 
summarizes the information from previous steps.
At each time step, the current input and the previous 
hidden state are combined to compute the new hidden 
state.
3. Activation Function: RNN units typically use an 
activation function, such as the hyperbolic tangent 
(tanh) or the rectified linear unit (ReLU), to introduce 
non-linearity to the hidden state computation.
The activation function introduces non-linear 
transformations to the inputs, allowing the RNN to learn 
complex patterns and relationships.
4. Weight Sharing: One of the key features of RNN is 
weight sharing, where the same set of weights and 
biases are used at each time step.
This allows the RNN to share information across 
different steps, enabling it to capture dependencies and 
patterns in the sequential data.
5. Training: During the training process, an RNN model 
is optimized to minimize a specific loss function, such 
as mean squared error (MSE) or cross-entropy loss.
The parameters of the network, including the weights 
and biases, are updated using gradient descent and 
backpropagation through time.
Backpropagation through time calculates gradients by 
propagating errors from the output back to the previous 
time steps.
6. Prediction: After training, an RNN model can be used 
to make predictions on new, unseen sequential data.
The model takes in the input sequence and iteratively 
processes each element, updating the hidden state and 
producing an output at each step.
The final output can be used for prediction or 
classification tasks.
RNNs are well-suited for tasks involving sequential 
data because they can capture and utilize information 
from previous steps. However, traditional RNNs can 
suffer from the vanishing gradient problem, where the 
gradients diminish as they propagate back in time, 
making it challenging to capture long-term 
dependencies. 
Overall, RNNs are widely used in various applications, 
including natural language processing, speech 
recognition, machine translation, and time series 
analysis, where the order and temporal dependencies of 
the data play a crucial role.
CNN
CNNs are primarily designed for image data, where 
they excel at capturing spatial patterns. However, CNNs 
can also be adapted for stock price prediction by treating 
the sequential data as a 1D grid, similar to a time series.
Here is a general approach to using CNNs for stock 
price prediction:
1. Data Representation: Convert the sequential stock 
price data into a 1D grid-like format, where the y-axis 
represents the stock price values, and the x-axis 
represents time steps.
Each data point in the sequence becomes a pixel in the 
grid, forming a 1D image.
2. Data Preprocessing: Preprocess the stock price data 
by handling missing values, normalizing the values, and 
splitting the data into training and testing sets.
3. Feature Extraction: Apply 1D convolutional layers to 
the stock price data grid.
The convolutional filters slide across the grid, capturing 
local patterns and extracting features related to price 
movements and trends.
The output of the convolutional layers is a set of feature 
maps.
4. Pooling: Apply pooling layers, such as max pooling, 
to reduce the dimensionality of the feature maps.
Pooling helps summarize the most important features 
while reducing noise and computational requirements.
5. Flattening and Fully Connected Layers: Flatten the 
pooled feature maps into a 1D vector.
Pass the flattened vector through fully connected layers 
to learn high-level representations and make 
predictions.
6. Training and Prediction: Train the CNN model on the 
training data using an appropriate loss function, such as 
mean squared error (MSE), and an optimizer.
Adjust the model's weights and biases using 
backpropagation and gradient descent.
Use the trained model to make predictions on the testing 
data.


It's important to note that while CNNs can be applied to 
stock price prediction, they may not capture all the 
nuanced patterns and dependencies in the data. Stock 
prices are influenced by a wide range of factors, 
including economic indicators, news events, and market 
sentiments, which may require additional information 
beyond the price sequence. Therefore, CNNs for stock 
price prediction is combined with other techniques 
LSTM (Long Short-Term Memory), to capture temporal 
dependencies and capture long-term patterns.


By treating the stock price sequence as a 1D grid and 
leveraging the convolutional and pooling operations of 
CNNs, it is possible to extract relevant features and 
learn patterns that can contribute to predicting future 
stock price movements.


LSTM AND RNN
LSTM and RNN models leverage their recurrent nature 
and the ability to capture dependencies across time steps 
to model and analyse sequential data effectively. The 
memory cell and gating mechanisms in LSTM help to 
overcome the vanishing gradient problem often 
encountered in traditional RNNs, allowing then to 
capture long-term dependencies more effectively.


LSTM AND CNN
We will explore the combination of LSTM and 
CNN models for stock price prediction. The CNN 
component will be responsible for feature extraction 
and LSTM component will capture the temporal 
dependencies within those features. This hybrid model 
will be trained on the training data and evaluated for its 
predictive performance


# RESULT AND DISCUSSION
In this study, we implemented the LSTM-RNN 
architecture for stock price prediction and evaluated its 
performance using a historical stock price dataset. The 
model was trained on daily stock prices from the S&P 
500 index, with the goal of predicting the closing price 
of the next day.
The LSTM-RNN model was trained using a sliding 
window approach, where the input sequence was a 
window of historical prices, and the output was the 
predicted price for the next day. We experimented with 
different window sizes and number of LSTM layers, and 
selected the best performing model based on validation 
loss.
The results showed that the LSTM-RNN model was 
able to predict stock prices with reasonable accuracy. 
The model achieved a mean squared error (MAE) of
0.067 and a root mean square error (RMSE) of 0.2612
on the test set, indicating that it was able to capture the 
general trends in the data.
However, there were also limitations to the LSTM-RNN 
model. One of the main challenges was feature 
selection, as the model relied on a set of pre-selected 
features to make predictions. Additionally, the model 
was sensitive to the length of the input sequence, with 
longer sequences leading to slower convergence and 
overfitting.
Overall, the LSTM-RNN architecture showed promise 
for stock price prediction, but also highlighted the 
importance of careful feature selection and 
hyperparameter tuning to achieve optimal performance.

After training and testing the LSTM-CNN model on the 
stock price dataset, we obtained the following results:
- Training loss: 0.0013
- Validation loss: 0.0012
- Test loss: 0.00088
We can observe that the LSTM-CNN model 
outperforms the LSTM-RNN model in terms of 
accuracy as well as loss. The training, validation, and 
test losses for the LSTM-CNN model are lower than 
those for the LSTM-RNN model. The model achieved a 
mean squared error (MSE) of 0.0008 and a root mean 
square error (RMSE) of 0.0296. This indicates that the 
LSTM-CNN model is better at predicting the stock 
prices than the LSTM-RNN model.
Additionally, we can visualize the predicted stock prices 
using the LSTM-CNN model and compare them with 
the actual stock prices. The visualization shows that the 
predicted stock prices follow the same trend as the 
actual stock prices, indicating that the model is 
accurately predicting the stock prices.
Overall, our results suggest that the LSTM-CNN model 
is a better choice for stock price prediction compared to 
the LSTM-RNN model. The LSTM-CNN model has 
higher accuracy and lower loss, making it a more 
reliable and accurate model for predicting stock prices.

# CONCLUSION
In conclusion, this study has explored the application of 
LSTM-RNN and LSTM-CNN architectures for stock 
price prediction. Both models were trained and tested on 
a historical stock price dataset, and their performance 
was evaluated based on various metrics such as mean 
absolute error (MAE) and root mean square error 
(RMSE).
The results showed that both LSTM-RNN and LSTM-CNN models were effective at predicting stock prices, 
with the LSTM-CNN model demonstrating slightly 
better accuracy. However, it is important to note that the 
performance of these models may vary depending on 
the specific dataset and features selected for analysis.
Overall, the study highlights the potential of deep 
learning techniques for financial analysis and 
forecasting, and demonstrates the effectiveness of 
LSTM-RNN and LSTM-CNN architectures for stock 
price prediction. These models can provide valuable 
insights for investors and financial analysts, enabling 
them to make more informed decisions and mitigate risk 
in the stock market.
However, there are still limitations and challenges 
associated with the use of deep learning models in 
finance, including the need for large and high-quality 
datasets, as well as the risk of overfitting and data 
snooping. Future research in this area should focus on 
addressing these challenges and developing more robust 
and effective models for financial analysis and 
forecasting

















