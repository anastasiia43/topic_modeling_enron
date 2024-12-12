# Topic Modeling with LSA, LDA, and Mallet on Enron Emails

This project implements topic modeling techniques using **LSA (Latent Semantic Analysis)**, **LDA (Latent Dirichlet Allocation)**, and **Mallet LDA**. The models are trained on the [**Enron Email Dataset**](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset), which consists of emails sent and received by employees of the Enron company.

## Data

The project uses the **Enron Email Dataset**, which contains over 500,000 emails exchanged between more than 150 employees of the Enron company. This dataset is widely used for research in natural language processing and topic modeling.
I use the text data from the emails to build models that uncover the key topics discussed in the emails, providing deeper insights into internal communications at the company.

## Code Structure

1. **Data Preprocessing**:
   - **Lemmatization** and **removal of stop words**.
   - Creation of a **corpus** and **dictionary** for further model training.

2. **LSA (Latent Semantic Analysis)**:
   - Parameters: number of topics (`num_topics`), number of iterations for iterative method (`power_iters`).
   - Using the **LsiModel** from the `gensim` library.
   - Model evaluation using **coherence** metrics such as **c_v** and **u_mass**.
   - Visualization of results using plots to compare coherence for different parameters.

3. **LDA (Latent Dirichlet Allocation)**:
   - Training the LDA model using **gensim.models.ldamodel.LdaModel**.
   - Parameters: number of topics (`num_topics`), number of passes (`passes`), number of iterations (`iterations`).
   - Using **Perplexity** as a metric to assess model quality.
   - Calculating **coherence** using the **c_v** method.

4. **Mallet LDA**:
   - Using the **Mallet** LDA model for improved efficiency.
   - Training parameters: number of topics (`num_topics`), number of iterations (`iterations`).
   - Evaluating coherence using the **c_v** method.
   - Visualization of topics with **pyLDAvis**.


## Key Functions

1. **`compute_coherence_UMass(corpus, dictionary, k, i)`**:
   - Computes the coherence of the LSA model with different topic and iteration parameters.

2. **`lda_model(num_topics, passes, iterations)`**:
   - Trains an LDA model with specified parameters.

3. **`compute_coherence_values(dictionary, corpus, texts, num_topics, iterations)`**:
   - Computes coherence for the LDA model trained using Mallet.

4. **`format_topics_sentences(ldamodel, corpus, texts)`**:
   - Formats output data, showing the dominant topics for each document along with the associated words.

