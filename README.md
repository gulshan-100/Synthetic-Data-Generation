
# Synthetic Data Generation for AI Systems

### Introduction

This project addresses the challenge of limited datasets in AI system training by implementing a synthetic data generation system. Focusing on creating artificial Amazon product reviews for supplements and vitamins, this system aims to produce diverse, realistic data for tasks such as intent detection, slot filling, and recommendation algorithms.

### Dataset Overview

* **Source**: Subset of Amazon reviews dataset
* **Category**: Supplements/Vitamins
* **Key Columns**:
    * 'text': Review content
    * 'title': Review title
    * 'rating': Product rating


## Methodology

### Libraries and Tools

* pandas, numpy: Data manipulation
* matplotlib: Visualization
* NLTK, spaCy: NLP tasks
* TensorFlow, Keras: Deep learning implementation

### Data Preprocessing

* Load dataset using pandas
* Remove null values
* Analyze word count distributions
* Standardize ratings
* Clean text data (lemmatization, stop word removal)
* Combine 'title' and 'text' into 'combined_text'

![Word Count of Reviews](https://github.com/gulshan-100/Synthetic-Data-Generation/blob/main/Images/download%20(1).png)




### Text Processing

* Tokenize text using Keras Tokenizer
* Generate n-gram sequences
* Pad sequences for uniform length

### Model Architecture

* Input: Text sequences and ratings
* Embedding: 100-dimensional
* Bidirectional LSTM: 128 units
* Attention: Self-attention mechanism
* Dense layers: 128 units, ReLU activation
* Output: Softmax activation for word prediction

### Training Process

* Loss: Categorical Cross-entropy
* Optimizer: Adam
* Batch size: 35
* Epochs: 80 (with early stopping)

### Text Generation

* Temperature controlled sampling
* Variable lengths, seed texts and temperatures

### Experimentation Details: Review Generation Model Evolution



#### 1. LSTM (Long Short-Term Memory)

**Implementation:** RNN layer with LSTM 

**Results:**
Improvement in text coherence
Still struggling with maintaining context over longer sequences and lot of repetitive text.

**Conclusion:**
LSTM showed promise but still fell short of generating high-quality, diverse reviews.

![LSTM Output](https://github.com/gulshan-100/Synthetic-Data-Generation/blob/main/Images/output-LSTM.png)



#### 2. Bidirectional LSTM
**Implementation:** Upgraded to Bidirectional LSTM
Added additional dense layers for better feature extraction

**Results:** Improved but BLEU score is quite low

**Conclusion:** Bidirectional LSTM showed significant improvement, but still room for enhancement.

#### 3. Attention Mechanism
**Implementation:** Integrated attention mechanism with Bidirectional LSTM. Used self-attention to focus on relevant parts of input sequence

**Results:**

* Substantial improvement in text quality and coherence
* Generated reviews showed better context awareness
*Significant increase in BLEU scores
![Output Attention40](https://github.com/gulshan-100/Synthetic-Data-Generation/blob/main/Images/output-attention40.png)


## Results and Analysis
### Training Outcomes

* Trained up to 80 epochs with early stopping
* Decreasing training loss indicates successful learning

![Training Performance](https://github.com/gulshan-100/Synthetic-Data-Generation/blob/main/Images/download%20(2).png)

### Generated Reviews

* 180 reviews generated
* Parameters varied:
    * Lengths: 10, 20, 30 words
    * Temperatures: 0.7, 1.0, 1.5
    * Multiple seed texts


### Quality Assessment

* Human Evaluation
* BLEU Score : Mean Blue score of generated text is 0.55 (approx)

## Key Questions Addressed

### Choice of Model Architecture

1. The LSTM with attention architecture was chosen for its ability to:

* Capture long term dependencies in text
* Focus on relevant input parts during word generation
* Maintain coherence and context in generated reviews
* Good BLEU score

2. Factors Considered in Dataset Generation

* Length variation (10, 20, 30 words)
* Temperature variation (0.7, 1.0, 1.5)
* Multiple seed texts
* Incorporation of product ratings

3. Measuring Synthetic Dataset Efficacy

* BLEU score: Comparison with real reviews
* Human evaluation: Manual coherence and realism assessment
* Downstream task performance: Comparative model training

4. Ensuring Inspired but Not Replicated Data

* Temperature based sampling for randomness
* Combining varied seed texts, lengths, and temperatures

5. Top Challenges Faced

* Generation of Quality reviews
* Integrating product ratings in generation
* Maintaining appropriate review structure
* Managing computational resource demands
