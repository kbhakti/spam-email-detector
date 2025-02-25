# Spam-Email-Detector

This project is a part of the AAI-500 course in the Applied Artificial Intelligence Program at the University of San Diego (USD). 

- Project Status: Completed

## Installation

To set up the project locally, follow these steps:

1. Clone the repository:

   ```bash
   https://github.com/aleena-gigi/spam-email-detector.git
   cd spam-email-detector
   
2. Install Dependencies:

   ```bash
   pip install -r requirements.txt

3. Run the Streamlit Application:
   
   ```bash
   streamlit run spam_classifier.py
   
## Objective

The main purpose of this project is to develop an automated email spam classifier. This project aims to build a Bayesian Spam Classifier using the Multinomial Naive Bayes algorithm. The classifier automatically detects and classifies emails as either spam or legitimate (ham) based on their content. This helps users reduce inbox clutter, prevent phishing atttacks and enhance cybersecurity by filtering our malicious emails.

This system can be used in email security, fraud detection and automated filtering systems used by businesses and individuals.

## Contributors
Teammates: Aleena Gigi, Bhakti Kanungo, Sabina George

## Methods Used

- **Natural Language Processing**: Tokenization and feature extraction using CountVectorizer
- **Machine Learning**: Trained a Multinomial Naïve Bayes model for spam detection
- **Data Manipulation**: Used for data cleaning, preprocessing and transformation
- **Data Visualization**: Performed EDA (Exploratory Data Analysis) using Matplotlib, Seaborn, and WordCloud
- **Cloud Computing**: Used Streamlit cloud community to host the model

## Technologies

- Python

## Project Description
In this project, we aim to build an automated email spam classifier utilizing Multinomial Naive Bayes algorithm. Based on the frequency of words in the text, this model will be able to classify an email as spam or ham. The project intends to improve email security and lower the risk of phishing attempts by identifying and filtering spam messages from emails.
   
### DataSet

The dataset used for this project is sourced from [Kaggle's Email Spam Classification](https://www.kaggle.com/datasets/mfaisalqureshi/spam-email) dataset. It contains a total of 5,572 emails, categorized as spam or ham (legitimate emails).
  - Number of Rows: 5,572 (each row represents an email)
  - Number of Columns: 2 (Category and Message)
  - Target Variable:
    - Spam (1) - Emails classified as spam
    - Ham (0) - Emails classified as legitimate

### Questions and Hypotheses
- Can a machine learning algorithm effectively distinguish between spam and ham emails?
- Which terms or phrases appear most frequently in spam emails?
- Does an email's length affect whether it's considered spam or ham?
- What is the performance of Naïve Bayes in comparison to other classification models?
  
### Methods Used
- Data Cleaning and preparation
    - removed missing values
    - removed duplicate values
    - The Category column was renamed to Label, and the Message column was renamed to Text for clarity
    - The Label column was mapped to numerical values where Ham = 0 and Spam = 1
 - Exploratory Data Analysis
     - Visualizing the distribution of spam and ham emails
     - word frequency analysis
     - message length analysis
 - Data Preprocessing
     - Textual data was transfromed into numerical features using CountVectorizer
 - Model Generation
     -  Using the Multinomial Naïve Bayes algorithm, the model was implemented and trained using the scikit-learn toolkit
 - Model Analysis
     - The effectiveness of the model was evaluated using key performance metrics including accuracy, precision, recall, F1-score and confusion matrix
 - Deployment
      - The model is deployed on [Streamlit Cloud Community](https://spam-email-detector-e4szmqkqpeueijb48inehs.streamlit.app)
  
 ### Challenges 
 - The dataset contains significantly more ham emails than spam, which may affect model performance
 - Minimizing incorrect classifications to prevent mislabeling legitimate emails as spam

## License
This project is licensed under the MIT License, allowing users to freely use, modify, and distribute the work with proper attribution. The full license details are available in the LICENSE file included in the project folder.

## Acknowledgement
We are grateful to present this project after successfully completing it. This project would not have been possible without the guidance, assistance, and suggestions of many individuals. We would like to express our deep sense of gratitude and indebtedness to each and every one who has helped us make this project a success.

We take this opportunity to express our deepest sense of gratitude and sincere thanks to everyone who helped us complete this work successfully. We express our sincere thanks to Dr. Zahid Hussain Wani, who has been our guide throughout this course. His mentorship, feedback, and support have been invaluable in shaping this project.


We would like to extend our gratitude to our professors and mentors for their guidance throughout this project. Special thanks to the contributors of the Kaggle dataset, as well as the scikit-learn and Streamlit communities for providing essential tools and resources.The successful development and implementation of this spam classification system has been made possible by their assistance.



