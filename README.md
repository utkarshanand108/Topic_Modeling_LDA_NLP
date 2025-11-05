# DataScienceCourse7_Assignment2
**Topic Modeling using Latent Dirichlet Allocation (LDA) — DS PGC Course 7 Assignment 2**

---

## 📘 Assignment Overview
This assignment focuses on **Topic Modeling**, an unsupervised Natural Language Processing (NLP) technique for uncovering hidden themes from a collection of text documents.  
You will implement **Latent Dirichlet Allocation (LDA)** to automatically extract topics from a small document dataset.

---

## 🧩 Tasks Summary

### 🧠 Task 1 — Data Exploration
- **Dataset:** `text_docs` (10 short documents)  
- Loaded using Pandas, inspected first few rows.  
- Verified no missing or duplicate entries.

```python
import pandas as pd
df = pd.read_excel("text_docs.xlsx")
print("Shape of dataset:", df.shape)
print("Unique documents:", df['document_id'].nunique())
print("Missing values per column:\n", df.isnull().sum())
```

**Output:**
```
Shape of dataset: (10, 2)
Unique documents: 10
Missing values per column:
 document_id    0
 text           0
```

---

### 🧹 Task 2 — Text Preprocessing
1. Converted all text to lowercase  
2. Removed punctuation and numbers  
3. Removed stopwords  
4. Tokenized text into word lists  

```python
import re, nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    return [t for t in tokens if t not in stop_words]

df['tokens'] = df['text'].apply(preprocess)
df[['text', 'tokens']].head()
```

**Example Output:**

| Document | Tokens |
|-----------|---------|
| 1 | [stock, market, experiencing, volatility, uncertainty] |
| 2 | [economy, growing, businesses, optimistic, future] |
| 3 | [climate, change, critical, issue, global, attention] |

---

### 🔢 Task 3 — Generate Topics using LDA
Created a **Document–Term Matrix** and trained an LDA model using Gensim.

```python
from gensim import corpora
from gensim.models import LdaModel

dictionary = corpora.Dictionary(df['tokens'])
corpus = [dictionary.doc2bow(text) for text in df['tokens']]
lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, random_state=42, passes=10)

for idx, topic in lda_model.print_topics(num_words=5):
    print(f"Topic {idx+1}: {topic}")
```

**Output Topics:**
```
Topic 1: attention, issue, critical, change, global
Topic 2: future, businesses, optimistic, growing, economy
Topic 3: industry, uncertainty, shaping, electric, volatility
Topic 4: renewable, energy, technologies, world, around
Topic 5: platforms, digital, ongoing, concern, become
```

---

### 📊 Interpretation
Each topic represents a coherent cluster of words:  
- **Topic 1:** Climate & Global Issues  
- **Topic 2:** Economy & Business Growth  
- **Topic 3:** Technology & Electric Vehicles  
- **Topic 4:** Renewable Energy Trends  
- **Topic 5:** Digital Transformation  

---

## 🧰 Tools & Libraries
| Category | Tools |
|-----------|-------|
| Language | Python 3 |
| Libraries | `pandas`, `nltk`, `gensim`, `re` |
| Model | Latent Dirichlet Allocation (LDA) |
| Environment | Jupyter Notebook / Google Colab |
| Dataset | text_docs.xlsx |

---

## 📂 Files Included
```
Topic Modeling using LDA NLP Project).pdf                # Problem statement
Topic Modeling using LDA (NLP Project) PDF.pdf           # Solution report (PDF)
Topic Modeling using LDA (NLP Project) Python Notebook.ipynb  # Jupyter Notebook
Topic Modeling using LDA (NLP Project) Python Script.py       # Python script version
```

---

## 🧭 How to Review
1. Open the `.ipynb` file for code + results.  
2. View `.py` for a clean script version.  
3. Open `.pdf` for final write-up and screenshots.  
4. Notebook includes topic words visualization and explanation.

---

## 👤 Author
**Utkarsh Anand**  
Data Science PGC Course 7 — Assignment 2  
Internshala Placement Guarantee Program
