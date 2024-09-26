# Synthetic Review Generation using NLP 

## 1. Introduction

This project aims to generate synthetic product reviews for supplement products using Natural Language Processing (NLP) and Deep Learning techniques. We use a dataset containing reviews and ratings, applying text preprocessing, tokenization, and building a custom model with an attention mechanism to generate reviews based on given seed texts. The generated reviews are influenced by ratings and temperature-based sampling, allowing for diverse outputs.

## 2. Dataset Overview

- **Source**: The dataset contains customer reviews related to supplements and vitamins.
- **Columns**:
  - `text`: The main content of the review.
  - `title`: The review title provided by the customer.
  - `rating`: The customer’s rating of the product (scaled between 1 to 5).
  
## 3. Objectives

- Clean and preprocess the text and title fields using NLP techniques.
- Generate synthetic reviews by leveraging a deep learning model, which includes:
  - Text tokenization and sequence generation.
  - Incorporation of attention mechanisms in the model.
  - Influence of user ratings on review generation.
  - Application of temperature-based sampling to control text diversity.
  
## 4. Preprocessing Steps

### 4.1 Text Cleaning

We use **spaCy** and **NLTK** for text cleaning:

- **Stopwords Removal**: Common words (like "the", "and", etc.) that don't contribute to sentiment or meaning are removed.
- **Lemmatization**: Words are reduced to their base or root forms to ensure consistency in tokenization.

The following functions and steps were applied:
- `clean_text`: Removes stopwords and lemmatizes the text.
- Both `title` and `text` columns are cleaned, then combined into a single column, `combined_text`.

### 4.2 Word Count Distribution

We plotted histograms showing the distribution of word counts for both the `text` and `title` columns to understand the length of reviews. These insights guided the sequence length selection for our model.
![Word Count of Reviews](https://github.com/gulshan-100/Synthetic-Data-Generation/blob/main/Images/download%20(1).png)

### 4.3 Rating Normalization

To ensure the `rating` field has a standard scale, we used **StandardScaler** from `sklearn` to normalize ratings to a mean of 0 and a standard deviation of 1.

## 5. Text Tokenization

We used Keras’ **Tokenizer** to convert text into a sequence of integers representing words, and fit the tokenizer on the `combined_text` column. This allows us to generate sequences of tokens from the text data.

- **Maximum Vocabulary Size**: Set to 9000, meaning only the top 9000 most frequent words in the dataset are considered.
- **Maximum Sequence Length**: Set to 80, meaning the longest review sequence will contain 80 tokens.

## 6. Model Architecture

### 6.1 Input Layers

- **Text Input**: The cleaned and tokenized text is passed through an **Embedding Layer**, followed by a **Bidirectional LSTM** for sequence learning.
- **Rating Input**: The normalized `rating` is provided as a separate input to guide the review generation.

### 6.2 Attention Mechanism

An attention layer is implemented to focus on important parts of the review during text generation. This is applied after the LSTM layer to capture the most relevant features for generating the next word in the sequence.

### 6.3 Model Architecture Overview

- **Embedding Layer**: Converts word tokens into dense vectors of fixed size (100 dimensions).
- **Bidirectional LSTM**: Processes the sequence in both forward and backward directions.
- **Attention Layer**: Focuses on important parts of the sequence, enhancing text generation.
- **Concatenation**: Combines attention output and the rating input.
- **Fully Connected Layers**: Dense layers with `ReLU` activation and `Dropout` for regularization.
- **Output Layer**: A softmax output over the vocabulary to predict the next word in the sequence.

### 6.4 Loss Function and Optimizer

- **Loss Function**: Categorical cross-entropy, as we’re predicting a word from a multi-class output.
- **Optimizer**: Adam optimizer for efficient training.

## 7. Training and Early Stopping

The model is trained using the `fit` function, with early stopping to prevent overfitting if the loss plateaus. The key parameters for training are:
- **Batch Size**: 35
- **Epochs**: 80 (Early stopping is monitored to stop training after 2 epochs of no improvement).
- **Loss Monitoring**: The loss is used as a metric to determine when to stop training.

![Training Performance](https://github.com/gulshan-100/Synthetic-Data-Generation/blob/main/Images/download%20(2).png)


## 8. Synthetic Review Generation

### 8.1 Temperature-based Sampling

Temperature-based sampling is used to control the diversity of the generated text:
- **Low temperature** (e.g., 0.7) generates more predictable and coherent reviews.
- **High temperature** (e.g., 1.5) generates more creative and diverse reviews, though with higher risk of incoherence.

### 8.2 Review Generation Process

1. **Seed Texts**: Predefined phrases such as "The supplement", "Fantastic experience", etc., are used as the initial input for generating reviews.
2. **Rating Influence**: The normalized rating is fed into the model along with the text to steer the generation process.
3. **Temperature Application**: The probabilities of the next word are adjusted using the temperature parameter to control creativity.
4. **Length**: The generated reviews are of different lengths (10,20,30)

### 8.3 Example Seed Texts
- "The supplement"
- "This product"
- "I really enjoyed"
- "Absolutely love"

Different review lengths (10, 20, 30 words) and temperature values (0.7, 1.0, 1.5) are used to diversify the output.

### 8.4 Review Generation Code
```python
def generate_review_with_temperature(model, tokenizer, seed_text, max_sequence_len, max_words, rating_value, num_words_to_generate=20, temperature=1.0):
    cleaned_seed_text = clean_text(seed_text)
    token_list = tokenizer.texts_to_sequences([cleaned_seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    
    generated_text = seed_text
    
    for _ in range(num_words_to_generate):
        predicted_probs = model.predict([token_list, np.array([[rating_value]])], verbose=0)[0]
        predicted_probs = apply_temperature(predicted_probs, temperature)
        predicted_word_index = np.random.choice(range(max_words), p=predicted_probs)
        
        if predicted_word_index == 0:
            break
        
        predicted_word = tokenizer.index_word.get(predicted_word_index, '')
        generated_text += ' ' + predicted_word
        
        token_list = np.append(token_list, [[predicted_word_index]], axis=1)
        token_list = pad_sequences([token_list[0]], maxlen=max_sequence_len - 1, padding='pre')
    
    return generated_text.strip()
```

### 9 KEY QUESTIONS
**1. Choice of Model Architecture**

The LSTM with attention architecture was chosen for its ability to:

- Capture long-term dependencies in text
- Focus on relevant input parts during word generation
- Maintain coherence and context in generated reviews

**2. Factors Considered in Dataset Generation**

- Length variation (10, 20, 30 words)
- Temperature variation (0.7, 1.0, 1.5)
- Multiple seed texts
- Incorporation of product ratings

**3. Measuring Synthetic Dataset Efficacy**

- BLEU score: Comparison with real reviews
- Human evaluation: Manual coherence and realism assessment
- Downstream task performance: Comparative model training

**4. Ensuring Inspired but Not Replicated Data**

- Temperature-based sampling for randomness
- Combining varied seed texts, lengths, and temperatures
- Attention mechanism for creative pattern combinations

**5. Top Challenges Faced**

- Integrating product ratings in generation
- Maintaining appropriate review structure
- Avoiding exact training data replication
- Managing computational resource demands
