# Movie-Reviews-Sentiment-Analysis  

This project performs sentiment analysis on movie reviews using **Natural Language Processing (NLP)** and **Machine Learning** techniques. It classifies each review as either **positive** or **negative** using a **Naive Bayes** model with **TF-IDF vectorization**.  

---

## Table of Contents  
- [Overview](#overview)  
- [Dataset](#dataset)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Model Explanation](#model-explanation)  
- [Results](#results)  
- [Technologies Used](#technologies-used)  
- [Contributing](#contributing)  
- [License](#license)  

---

## Overview  
This is a **Text Classification** project under **Supervised Machine Learning**, where the goal is to predict the **sentiment polarity** (positive or negative) of movie reviews using **Natural Language Processing**. The model is trained on the IMDB dataset and uses a pipeline combining **TF-IDF Vectorizer** and **Multinomial Naive Bayes**.

---

## Dataset  
- **Source:** [IMDB Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)  
- **Size:** 50,000 movie reviews (25,000 positive and 25,000 negative)  
- **Attributes:**  
  - *Review:* Full text of the movie review  
  - *Sentiment:* Sentiment label — `positive` or `negative`  

---

## Installation  
1. **Clone the repository:**  
   ```bash
   git clone https://github.com/your-username/Movie-Reviews-Sentiment-Analysis.git
   cd Movie-Reviews-Sentiment-Analysis
   ```
2. **Install dependencies:**  
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage  
- Run the script to train and test the model.  
- The trained model will be saved as `sentiment_model.pkl`.  
- You can use this model to predict sentiment for any new review.

```python
import pickle

with open("sentiment_model.pkl", "rb") as file:
    model = pickle.load(file)

sample_review = ["This movie was a masterpiece with brilliant acting."]
prediction = model.predict(sample_review)
print("Positive" if prediction[0] == 1 else "Negative")
```

---

## Model Explanation  
- **Algorithm Used:** Multinomial Naive Bayes  
- **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency)  
- **Preprocessing:**  
  - Lowercasing  
  - Stopword removal (NLTK)  
  - Label encoding (positive → 1, negative → 0)  

---

## Results  
- The model achieved **~(your_accuracy_here)** accuracy on the test dataset.  
- Example:  
  - **Input:** "The plot was boring and predictable."  
  - **Prediction:** Negative

---

## Technologies Used  
- Python  
- pandas, scikit-learn, nltk  
- TfidfVectorizer  
- MultinomialNB  
- Pickle (model saving)

---


## License  
This project is licensed under the **MIT License**.
