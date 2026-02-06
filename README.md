# 🛒 Flipkart Review Sentiment Analysis Project

_Analyzing customer sentiment from Flipkart product reviews using NLP, Machine Learning, and Streamlit._

The **Flipkart Review Sentiment Analysis Project** focuses on understanding customer opinions by analyzing product reviews using **Natural Language Processing (NLP)** and **Machine Learning**.

This project classifies reviews into **Positive** or **Negative** sentiments and provides real-time predictions through an interactive **Streamlit web application**, helping businesses quickly assess customer feedback at scale.

The primary objectives of this project are to:
- Analyze customer sentiment from Flipkart product reviews  
- Automatically classify reviews as **Positive** or **Negative**  
- Reduce manual effort in feedback analysis  
- Support data-driven decision-making for product and service improvement  

- **Dataset Name:** Flipkart Product Reviews Dataset  
- **Records:** 8508 reviews  
- **Columns:** Review Text, Sentiment Label, Product Metadata  
- **Source:** Public e-commerce review dataset  
- **Target Variable:** Sentiment (Positive / Negative)

- **Python** – Core programming language  
- **Pandas & NumPy** – Data handling and processing  
- **NLTK** – Text preprocessing (tokenization, stopwords, lemmatization)  
- **Scikit-learn** – TF-IDF, ML models, evaluation  
- **Streamlit** – Interactive web application  
- **Pickle** – Model and vectorizer persistence  

Sentiment analysis project/
│
├── app/
│   └── app.py                  # Streamlit web app
├── data/
│   └── flipkart_reviews.csv    # Dataset
├── model_building/
│   └── model_building.py       # Model training & evaluation
├── pkl/
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
├── requirements.txt
├── README.md
└── .gitignore

The following preprocessing steps were applied:

- Removal of missing and duplicate reviews  
- Conversion of text to lowercase  
- Removal of special characters and punctuation  
- Tokenization of text  
- Stopword removal  
- Lemmatization to normalize words  

- **Feature Extraction:**  
  - TF-IDF Vectorization (Unigrams + Bigrams)

- **Models Trained & Compared:**  
  - Logistic Regression  
  - Naive Bayes  
  - Linear Support Vector Machine (SVM)  
  - Random Forest  

- **Evaluation Metric:**  
  - F1 Score  

- **Final Model Selected:**  
  - **Linear SVM** (best performance)
    
The Streamlit web application includes:

- Text area for entering product reviews  
- Center-aligned prediction button  
- Emoji-based sentiment output  
  - 😄😊🎉 Positive Review  
  - 😢😞💔 Negative Review  
- Clean, responsive, and user-friendly interface  

1. Linear SVM performed best among all tested models  
2. TF-IDF with bigrams improved sentiment detection accuracy  
3. Short and clear reviews are classified more confidently  
4. NLP preprocessing significantly boosts model performance  

- Extend model to support **Neutral sentiment**  
- Add **confidence score** for predictions  
- Use **deep learning models** for further improvement  
- Integrate sentiment insights into business dashboards  

This project enables organizations to:
- Automatically analyze large volumes of customer feedback  
- Improve product quality using sentiment trends  
- Enhance customer experience and satisfaction  
- Save time and cost compared to manual review analysis  

### 📊 Conclusion
- NLP and ML can effectively classify customer sentiment  
- TF-IDF + Linear SVM provides strong baseline performance  
- Real-time sentiment analysis is achievable with Streamlit  

### 🧠 Learnings
- Built an end-to-end NLP pipeline  
- Gained hands-on experience with text preprocessing  
- Learned model comparison and evaluation techniques  
- Designed a deployable ML web application  


1. **Clone the repository**
```bash
https://github.com/Technology-Manasi/GENAI.git

2. **Install dependencies**

pip install -r requirements.txt

3. **Train the model**

python model_building/model_building.py


4. **Run the Streamlit app**

streamlit run app/app.py

**Manasi Rawool**
IT student

Special thanks to **Innomatics Research Labs** for providing hands-on industry-oriented training.
Heartfelt gratitude to my mentors for their guidance and continuous support throughout this project.

