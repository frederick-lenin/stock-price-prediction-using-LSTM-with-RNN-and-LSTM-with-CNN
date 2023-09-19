# stock-price-prediction-using-LSTM-with-RNN-and-LSTM-with-CNN

## This is my final year project

ABSTRACT:
In the current financial sector, predicting stock prices is a key activity, and most of the prediction algorithms are typically based on machine learning and deep learning algorithms. Nonstationary time series data are challenging for these algorithms to handle, though. For predicting stock price, deep learning methods including convolutional neural networks (CNNs), recurrent neural networks (RNNs), and long short-term memory (LSTM) networks have been extensively used. Because they can store historical data, LSTMs are extremely effective prediction methods. We suggested a fresh deep learning prediction technique. This study suggests a technique for forecasting changes in stock prices using recently updated stock prices. We evaluate variations in accuracy while concatenating two alternative top algorithms, LSTM-RNN and LSTM-CNN.
Keywords – Stock price; LSTM; RNN; CNN; deep learning

INTRODUCTION
1.1 MOTIVATION OF WORK
The motivation behind using LSTM (Long Short-Term Memory) and RNN (Recurrent Neural Network) models for stock price prediction, as well as combining LSTM with CNN (Convolutional Neural Network), is to 
leverage the power of deep learning to capture and learn from the sequential and temporal patterns present in stock market data. Here are a few reasons why these models are commonly used for stock price 
prediction: 
1. Sequential data analysis: Stock market data is inherently sequential, where the order of data points matters. LSTM and RNN models are designed to effectively model and capture dependencies between past and present data points, making them suitable for analysing and predicting stock price movements.
2. Long-term dependencies: LSTM, a variant of RNN, is specifically designed to address the vanishing gradient problem associated with traditional RNNs. It can retain information for longer periods, allowing it to capture longterm dependencies in the data. This is crucial for stock price prediction, as past prices and trends can have a significant impact on future prices.
3. Non-linear patterns: Stock market data exhibits complex and nonlinear patterns, which cannot be easily captured by traditional statistical models. Deep learning models, such as LSTM and CNN, have the ability to learn complex patterns and relationships from the data, enabling them to make more accurate predictions.
4. Feature extraction: CNNs are widely used in computer vision tasks for their ability to automatically extract relevant features from raw data. Similarly, in stock market prediction, CNNs can be used to extract relevant features from historical price data, allowing the LSTM model to focus on capturing temporal dependencies.
5. Handling high-dimensional data: Stock market data often consists of multiple variables or features, such as price, volume, and technical indicators. LSTM and CNN models can handle high-dimensional data efficiently, allowing them to capture and learn from the multi-dimensional relationships in the data.
By combining LSTM and RNN models with CNNs, we can leverage the strengths of both architectures. The CNN can extract meaningful features from the input data, and the LSTM or RNN can effectively capture the temporal dependencies within those features. This combined approach has shown promising results in capturing complex patterns and improving the accuracy of stock price predictions. Overall, the motivation behind using LSTM, RNN, and LSTM-CNN models for stock price prediction is to leverage their ability to capture sequential patterns, handle high-dimensional data, learn complex relationships, and make accurate predictions based on historical price data.
1.2 PROBLEM STATEMENT
The objective of this project is to develop a predictive model for stock price movements based on historical stock market data. The problem can be defined as follows:
1. Data Collection: Historical stock market data needs to be collected for the target stock(s) from reliable sources. This data should cover a significant time period to capture various market conditions and trends.
2. Feature Selection: Relevant features, such as price-related attributes (opening, closing, high, low), trading volume, and technical indicators, need to be identified and extracted from the collected data. The selection of features should be based on their importance and relevance to stock price movements.
3. Data Pre-processing: The collected data needs to be pre-processed to handle missing values, outliers, and any inconsistencies. Additionally, normalization or scaling techniques may be applied to ensure that the features are on a comparable scale and prevent bias in the model.
4. Time Series Analysis: Stock market data is inherently sequential, with the order of observations playing a crucial role. The model should be designed to capture the temporal dependencies and patterns present in the data.
5. Model Training: Various deep learning algorithms can be explored for stock price prediction, such as LSTM (Long Short-Term Memory), RNN (Recurrent Neural Network), or other regression models. The selected model should be trained on the pre-processed data, optimizing its parameters to minimize the prediction error.
6. Model Evaluation: The trained model needs to be evaluated using suitable evaluation metrics, such as mean squared error (MSE), mean absolute error (MAE), or accuracy. The evaluation should be performed on a separate testing dataset that was not used during the training phase to assess the model's generalization ability.
7. Predictive Performance: The effectiveness of the model will be assessed based on its ability to accurately predict the future price movements of the target stock(s). The model should demonstrate a reasonable level of accuracy and reliability in capturing the market trends.
By addressing the problem statement, the aim is to develop a robust and accurate predictive model that can assist investors, traders, and financial professionals in making informed decisions related to stock market investments. The model should provide valuable insights into the potential direction of stock prices, enabling users to optimize their trading strategies and mitigate investment risks.

LITERATURE SURVEY
2.1 GENERAL
Stock price prediction is a challenging task in the financial industry, and the application of deep learning techniques such as recurrent neural networks (RNNs) and long short-term memory (LSTM) models has gained significant attention. In this literature survey, we explore ten noble studies that have utilized LSTM and RNN models for stock price prediction, highlighting their methodologies, datasets, and performance metrices.
2.2 RELATED WORK
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





















