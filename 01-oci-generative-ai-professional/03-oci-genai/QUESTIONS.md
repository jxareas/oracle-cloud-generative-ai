# Skill Check - OCI GenAI Service

1. What is a distinctive feature of GPUs in Dedicated AI Clusters used for generative AI tasks?
    - [ ] GPUs are used exclusively for storing large datasets, not for computation.
    - [ ] Each customer's GPUs are connected via a public internet network for ease of access.
    - [ ] GPUs are shared with other customers to maximize resource utilization.
    - [x] GPUs allocated for a customer's generative AI tasks are isolated from other GPUs.

      **Explanation**: The GPUs allocated for a customer’s generative AI tasks are isolated from other GPUs to maintain
      the security and privacy of customer data and workloads.

2. What happens if a period (.) is used as a stop sequence in text generation?
    - [ ] The model stops generating text after it reaches the end of the current paragraph.
    - [ ] The model generates additional sentences to complete the paragraph.
    - [ ] The model stops generating text once it reaches the end of the first sentence, even if the token limit is much
      higher.
    - [ ] The model ignores periods and continues generating text until it reaches the token limit.

      **Explanation**: Stop sequences, in the context of text generation, are special tokens or symbols used to signal
      the end of the generated text. These sequences serve as markers for the model to halt its generation process.
      Common stop sequences include punctuation marks such as periods (.), question marks (?), and exclamation
      marks (!), as they typically denote the end of sentences in natural language.

3. What is the purpose of embeddings in natural language processing?
    - [ ] To increase the complexity and size of text data
    - [ ] To translate text into a different language
    - [x] To create numerical representations of text that capture the meaning and relationships between words or
      phrases
    - [ ] To compress text data into smaller files for storage

      **Explanation**: Embeddings map words or text on to a continuous vector space where similar words are located
      close to each other. This allows NLP models to capture semantic relationships between words, such as synonyms or
      related concepts. For example, in a well-trained embedding space, the vectors for "king" and "queen" would be
      closer to each other than to unrelated words like "car" or "tree." Embeddings also provide a dense,
      low-dimensional representation of words compared to traditional one-hot encodings. This makes them more efficient
      and effective as input features for machine learning models, reducing the dimensionality of the input space, and
      improving computational efficiency.

4. What is the main advantage of using few-shot model prompting to customize a Large Language Model (LLM)?
    - [ ] It allows the LLM to access a larger dataset.
    - [x] It provides examples in the prompt to guide the LLM to better performance with no training cost.
    - [ ] It eliminates the need for any training or computational resources.
    - [ ] It significantly reduces the latency for each model request.

      **Explanation**: The main advantage of using few-shot model prompting to customize a Large Language Model (LLM) is
      its ability to adapt the model quickly and effectively to new tasks or domains with only a small amount of
      training data. Instead of retraining the entire model from scratch, which can be time-consuming and
      resource-intensive, few-shot prompting leverages the model's pre-existing knowledge.

5. What is the purpose of frequency penalties in language model outputs?
    - [ ] To ensure tokens that appear frequently are used more often
    - [x] To penalize tokens that have already appeared, based on the number of times they've been used
    - [ ] To randomly penalize some tokens to increase the diversity of the text
    - [ ] To reward the tokens that have never appeared in the text

      **Explanation**: Frequency penalties in language model outputs aim to discourage the repetition of tokens that
      have already appeared in the generated text. When generating text, language models may tend to produce repetitive
      phrases or words, which can lead to less diverse and less interesting outputs. By applying frequency penalties,
      tokens that have been used multiple times are penalized, reducing the likelihood of their repetition in subsequent
      generations.



