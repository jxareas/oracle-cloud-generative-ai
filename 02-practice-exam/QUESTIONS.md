# Oracle Generative AI Professional Practice Exam

1. A data science team is fine-tuning multiple models using the Oracle Generative AI service. They select the
   `cohere.command-r-08-2024` base model and fine-tune it on three different datasets for three separate tasks. They
   plan
   to use the same fine-tuning AI cluster for all models.
   What is the total number of units provisioned for the cluster?
    - [ ] 1
    - [ ] 6
    - [x] 8
    - [ ] 2

2. How is the totalTrainingSteps parameter calculated during fine-tuning in OCI Generative AI?
    - [x] totalTrainingSteps = (totalTrainingEpochs * size(trainingDataset)) / trainingBatchSize
    - [ ] totalTrainingSteps = (totalTrainingEpochs + size(trainingDataset)) * trainingBatchSize
    - [ ] totalTrainingSteps = (totalTrainingEpochs * trainingBatchSize) / size(trainingDataset)
    - [ ] totalTrainingSteps = (size(trainingDataset) * trainingBatchSize) / totalTrainingEpochs

3. How does the temperature setting in a decoding algorithm influence the probability distribution over the vocabulary?
    - [x] Increasing temperature flattens the distribution, allowing for more varied word choices.
    - [ ] Increasing temperature removes the impact of the most likely word.
    - [ ] Decreasing temperature broadens the distribution, making less likely words more probable.
    - [ ] Temperature has no effect on the probability distribution; it only changes the speed of decoding.

4. How can you affect the probability distribution over the vocabulary of a Large Language Model (LLM)?
    - [x] By using techniques like prompting and training
    - [ ] By modifying the model's training data
    - [ ] By restricting the vocabulary used in the model
    - [ ] By adjusting the token size during the training phase

5. When specifying a data source, what does enabling multi-modal parsing do?
    - [ ] Parses and converts non-supported file formats into supported ones
    - [ ] Merges multiple data sources into a single knowledge base after parsing the files
    - [x] Parses and includes information from charts and graphs in the documents
    - [ ] Automatically tags files and folders in the bucket

6. How does a presence penalty function when using GCI Generative AI chat models?
    - [ ] It applies a penalty only if the token has appeared more than twice.
    - [ ] It only penalizes tokens that have never appeared in the text before.
    - [ ] It penalizes all tokens equally, regardless of how often they have appeared.
    - [x] It penalizes a token each time it appears after the first occurrence.

7. What advantage does fine-tuning offer in terms of improving model efficiency?
    - [ ] It eliminates the need for annotated data during training.
    - [ ] It improves the model's understanding of human preferences.
    - [ ] It increases the model's context window size.
    - [x] It reduces the number of tokens needed for model performance.

8. A marketing team is using Oracle's Generative AI service to create promotional content. They want to generate
   consistent responses for the same prompt across multiple runs to ensure uniformity in their messaging. They notice
   that the responses vary each time they run the model, despite keeping the prompt and other parameters the same.
    ```python
    chat_request.seed = None
    chat_request.temperature = 0
    chat_request.frequence_penalty = 1
    chat_request.top_p = 0.75
    ```
   Which parameter should they modify to ensure identical outputs for the same input?
    - [ ] frequency_penalty
    - [ ] top_p
    - [x] seed
    - [ ] temperature

9. A company is using a model in the OCI Generative AI service for text summarization. They receive a notification
   stating that the model has been deprecated.
   What action should the company take to ensure continuity in their application?
    - [x] The company can continue using the model but should start planning to migrate to another model before it is
      retired.
    - [ ] The company should ignore the notification as deprecated models remain available indefinitely.
    - [ ] The company must immediately stop using the model because it is no longer available and start using the newer
      model.
    - [ ] The company can request an extension to continue using the model after it is retired.

10. You want to build an LLM application that can connect application components easily and allow for component
    replacement in a declarative manner.
    What approach would you take?
    - [ ] Use agents.
    - [ ] Use Python classes like LLMChain.
    - [ ] Use prompts.
    - [x] Use LangChain Expression Language (LCEL).

11. What is the purpose of the **VECTOR** field in the Oracle Database `23ai` table for Generative AI Agents?
    - [ ] To store the URL references for the documents
    - [x] To store the embeddings generated from the `BODY` content
    - [ ] To assign a unique identifier `DOCID` to each document
    - [ ] To store the document `TITLE`

12. A startup is using Oracle Generative AI's on-demand inferencing for a chatbot. The chatbot processes user queries
    and generates responses dynamically. One user enters a **200-character prompt**, and the model generates a *
    *500-character response**.

    How many transactions will be billed for this inference call?
    - [ ] 200 transactions
    - [ ] 500 transactions
    - [x] 700 transactions
    - [ ] 1 transaction per API call, regardless of length

13. What is the role of the inputs parameter in the given code snippet?
    ```python
    inputs = [
      "Learn about the Employee Stock Purchase Plan",
      "Reassign timecard approvals during leave",
      "View my payslip online",
    ]
    embed_text_detail.inputs = inputs
    ```
    - [ ] It controls the maximum number of embeddings the model can generate.
    - [ ] It sets the output format for the embeddings.
    - [ ] It provides metadata about the embedding process.
    - [x] It specifies the text data that will be converted into embeddings.

14. When activating content moderation in OCI Generative AI Agents, which of these can you specify?
    - [ ] The maximum file size for input data
    - [x] Whether moderation applies to user prompts, generated responses, or both
    - [ ] The threshold for language complexity in responses
    - [ ] The type of vector search used for retrieval

15. How many numerical values are generated for each input phrase when using the cohere.embed-english-light-v3.0
    embedding model?
    - [x] 384
    - [ ] 512
    - [ ] 256
    - [ ] 1024

16. Which statement is true about the "Top p" parameter of OCI Generative AI chat models?
    - [x] "Top p" limits token selection based on the sum of their probabilities.
    - [ ] "Top k" selects tokens from the "top k" tokens sorted by probability.
    - [ ] "Top p" assigns penalties to frequently occurring tokens.
    - [ ] "Top p" determines the maximum number of tokens per response.

17. In an OCI Generative AI chat model, which of these parameter settings is most likely to induce hallucinations and
    factually incorrect information?
    - [ ] temperature = 0.2, top_p = 0.6, and frequency_penalty = 0.8
    - [x] temperature = 0.9, top_p = 0.8, and frequency_penalty = 0.1
    - [ ] temperature = 0.5, top_p = 0.9, and frequency_penalty = 0.5
    - [ ] temperature = 0.0, top_p = 0.7, and frequency_penalty = 1.0

18. When is fine-tuning an appropriate method for customizing an LLM?
    - [ ] When the LLM requires access to the latest data for generating outputs
    - [ ] When you want to optimize the model without any instructions
    - [x] When the LLM does not perform well on a particular task and the data required to adapt the LLM is too large
      for prompt engineering
    - [ ] When the LLM already understands the topics necessary for text generation

19. In the context of RAG, how might the concept of Groundedness differ from that of Answer Relevance?
    - [ ] Groundedness focuses on data integrity, while Answer Relevance emphasizes lexical diversity.
    - [ ] Groundedness measures relevance to the user query, while Answer Relevance evaluates data integrity.
    - [ ] Groundedness refers to contextual alignment, while Answer Relevance deals with syntactic accuracy.
    - [x] Groundedness pertains to factual correctness, while Answer Relevance concerns query relevance.

20. A company is using a Generative AI model to assist customer support agents by answering product-related queries.

    Customer query: "What are the supported features of your new smart watch?"
    Generative AI model response: "The smart watch includes ECG monitoring, blood sugar tracking, and solar charging."

    Upon review of this response, the company notes that blood sugar tracking and solar charging are not actual features
    of their smart watch. These details were not part of the company's product documentation or database.

    What is the most likely cause of this model behavior?
    - [ ] The model was unable to access the company's database, so it defaulted to guessing feature sets based on
      similar products.
    - [ ] The model encountered a prompt that was too ambiguous, leading to random outputs.
    - [ ] The model is overfitting to specific details from unrelated training data, causing inaccuracies.
    - [x] The model is hallucinating, confidently generating responses that are not grounded in factual or provided
      data.

21. Accuracy in vector databases contributes to the effectiveness of LLMs by preserving a specific type of relationship.
    What is the nature of these relationships, and why are they crucial for language models?
    - [ ] Hierarchical relationships, and they are important for structuring database queries
    - [x] Semantic relationships, and they are crucial for understanding context and generating precise language
    - [ ] Temporal relationships, and they are necessary for predicting future linguistic trends
    - [ ] Linear relationships, and they simply simplify the modeling process

22. When does a chain typically interact with memory in a run within the LangChain framework?
    - [x] After user input but before chain execution, and again after core logic but before output
    - [ ] Before user input and after chain execution
    - [ ] Only after the output has been generated
    - [ ] Continuously throughout the entire chain execution process

23. In the context of generating text with a Large Language Model (LLM), what does the process of greedy decoding
    entail?
    - [ ] Selecting a random word from the entire vocabulary at each step
    - [ ] Picking a word based on its position in a sentence structure
    - [ ] Using a weighted random selection based on a modulated distribution
    - [x] Choosing the word with the highest probability at each step of decoding

24. What is a key effect of deleting a data source used by an agent in Generative AI Agents?
    - [ ] The agent starts generating responses based on pretrained data.
    - [x] The agent no longer answers questions related to the deleted source.
    - [ ] The agent automatically ingests data from a different source.
    - [ ] The agent stops running completely.

25. In the simplified workflow for managing and querying vector data, what is the role of indexing?
    - [ ] Compressing vector data for minimized storage usage
    - [x] Mapping vectors to a data structure for faster searching, enabling efficient retrieval
    - [ ] Converting vectors into a non-indexed format for easier retrieval
    - [ ] Categorizing vectors based on their originating data type (text, images, audio)
      I understand. Here is the parsing for the question in the last image you provided, **Question 26**, in the
      requested format:

26. What is the role of the OnDemandServingMode in the following code snippet?
    ```python
    chat_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(model_id="ocid1.generativeaimodel.oc1.eu-frankfurt-1.xxxxxxxxxxxxxxxxxxxx")
    ```
    - [ ] It defines the retry strategy for handling failures during model inference.
    - [x] It specifies that the Generative AI model should serve requests only on demand, rather than continuously.
    - [ ] It configures the model to use batch processing for requests.
    - [ ] It initializes the model with the default configuration profile for inference.

27. endpoint = "https://inference.generativeai.eu-frankfurt-1.oraclecloud.com"

    What is the purpose of this endpoint variable in the code?
    - [ ] It stores the OCI API key required for authentication.
    - [ ] It sets the retry strategy for the inference client.
    - [ ] It specifies the availability domain where the OCI Generative AI model is hosted, ensuring inference happens
      in the correct region.
    - [x] It defines the URL of the OCI Generative AI inference service.

28. What must be done before you can delete a knowledge base in Generative AI Agents?
    - [x] Delete the data sources and agents using that knowledge base.
    - [ ] Archive the knowledge base for future use.
    - [ ] Disconnect the database tool connection.
    - [ ] Reassign the knowledge base to a different agent.
29. A machine learning engineer is exploring T-Few fine-tuning to efficiently adapt a Large Language Model (LLM) for a
    specialized NLP task. They want to understand how T-Few fine-tuning modifies the model compared to standard
    fine-tuning techniques.

    Which of these best describes the characteristic of T-Few fine-tuning for LLMs?
    - [ ] It does not update any weights but restructures the model architecture.
    - [ ] It updates all the weights of the model uniformly.
    - [x] It selectively updates only a fraction of the model's weights.
    - [ ] It increases the training time as compared to Vanilla fine-tuning.

30. A data scientist is training a machine learning model to predict customer purchase behavior. After each training
    epoch, they analyze the loss metric reported by the model to evaluate its performance. They notice that the loss
    value is decreasing steadily over time.

    What does the loss metric indicate about the model's predictions in this scenario?
    - [x] Loss quantifies how far the model's predictions deviate from the actual values, indicating how wrong the
      predictions are.
    - [ ] Loss reflects the quality of predictions and should increase as the model improves.
    - [ ] Loss only evaluates the accuracy of correct predictions, ignoring the impact of incorrect predictions.
    - [ ] Loss measures the total number of predictions made by the model during training.

31. What happens to chat data and retrieved context after the session ends in OCI Generative AI Agents?
    - [ ] They are archived for audit purposes.
    - [ ] They are stored in isolation for future customer usage, ensuring maximum security but not used for training.
    - [x] They are permanently deleted and not retained.
    - [ ] They are stored for training the Large Language Models (LLMs).
32. A data scientist is exploring Retrieval-Augmented Generation (RAG) for a natural language processing project.

    Which statement is true about RAG?
    - [x] It is non-parametric and can theoretically answer questions about any corpus.
    - [ ] It is not suitable for fact-checking because of high hallucination occurrences.
    - [ ] It is solely used in QA-based scenarios.
    - [ ] It is primarily parametric and requires a different model for each corpus.

33. Which phase of the RAG pipeline includes loading, splitting, and embedding of documents?
    - [ ] Retrieval
    - [ ] Evaluation
    - [ ] Generation
    - [x] Ingestion

34. What does the OCI Generative AI service offer to users?
    - [ ] Only pretrained LLMs with customization options
    - [ ] A service requiring users to share GPUs for deploying LLMs
    - [ ] A limited platform that supports chat-based LLMs without hosting capabilities
    - [x] Fully managed LLMs along with the ability to create custom fine-tuned models

35. What happens when this line of code is executed?
    ```python
    embed_text_response = generative_ai_inference_client.embed_text(embed_text_detail)
    ```
    - [x] It sends a request to the OCI Generative AI service to generate an embedding for the input text.
    - [ ] It processes and configures the OCI profile settings for the inference session.
    - [ ] It initiates a connection to OCI and authenticates using the user's credentials.
    - [ ] It initializes a pretrained OCI Generative AI model for use in the session.

36. What does a cosine distance of 0 indicate about the relationship between two embeddings?
    - [ ] They are unrelated.
    - [x] They are similar in direction.
    - [ ] They have the same magnitude.
    - [ ] They are completely dissimilar.

37. A researcher is exploring generative models for various tasks. While diffusion models have shown excellent results
    in generating high-quality images, they encounter significant challenges in adapting these models for text.

    What is the primary reason why diffusion models are difficult to apply to text generation tasks?
    - [ ] Because diffusion models can only produce images
    - [ ] Because text generation does not require complex models
    - [ ] Because text is not categorical
    - [x] Because text representation is categorical, unlike images

38. Which statement regarding fine-tuning and Parameter-Efficient Fine-Tuning (PEFT) is correct?
    - [x] Fine-tuning requires training the entire model on new data, often leading to substantial computational costs,
      whereas PEFT involves updating only a small subset of parameters, minimizing computational requirements and data
      needs.
    - [ ] Fine-tuning and PEFT do not involve model modification; they differ only in the type of data used for
      training, with fine-tuning requiring labeled data and PEFT utilizing unlabeled data.
    - [ ] Both fine-tuning and PEFT require the model to be trained from scratch on new data, making them equally data
      and computationally intensive.
    - [ ] PEFT requires replacing the entire model architecture with a new one designed specifically for the new task,
      making it significantly more data-intensive than fine-tuning.

39. In which scenario is soft prompting more appropriate compared to other training styles?
    - [ ] When the model requires continued pretraining on unlabeled data
    - [ ] When there is a significant amount of labeled, task-specific data available
    - [x] When there is a need to add learnable parameters to a LLM without task-specific training
    - [ ] When the model needs to be adapted to perform well in a domain it was not originally trained on

40. Which of these is NOT a supported knowledge base data type for OCI Generative AI Agents?
    - [ ] Oracle Database 23ai vector search
    - [x] Custom-built file systems
    - [ ] OCI Object Storage files with text and PDFs
    - [ ] OCI Search with OpenSearch
41. You are debugging and testing an OCI Generative AI chat model.

    What is the model behavior if you don't provide a value for the seed parameter?
    - [ ] The model generates responses deterministically.
    - [x] The model gives diverse responses.
    - [ ] The model assigns a default seed value of 9999.
    - [ ] The model restricts the maximum number of tokens that can be generated.

42. What happens to the status of an endpoint after initiating a move to a different compartment?
    - [x] The status changes to Updating during the move and returns to Active after completion.
    - [ ] The status remains Active throughout the move.
    - [ ] The endpoint is deleted and recreated in the new compartment.
    - [ ] The endpoint becomes inactive permanently, and you need to create a new endpoint.

43. Which fine-tuning methods are supported by the `cohere.command-r-08-2024` model in OCI Generative AI?
    - [ ] T-Few and Vanilla
    - [x] T-Few and LoRA
    - [ ] T-Few, LoRA, and Vanilla
    - [ ] LoRA and Vanilla

44. In which phase of the RAG pipeline are additional context and user query used by LLMs to respond to the user?
    - [x] Generation
    - [ ] Retrieval
    - [ ] Ingestion
    - [ ] Evaluation

45. What is the destination port range that must be specified in the subnet's ingress rule for an Oracle Database in OCI
    Generative AI Agents?
    - [ ] 3306-3307
    - [x] 1521-1522
    - [ ] 1433-1434
    - [ ] 8080-8081

46. You are developing an application that displays a house image along with its related details. Assume that you are
    using Oracle Database 23ai.

    Which data type should be used to store the embeddings of the images in a database column?
    - [ ] `Float32`
    - [ ] `Double`
    - [ ] `INT`
    - [x] `VECTOR`

47. Which component of Retrieval-Augmented Generation (RAG) evaluates and prioritizes the information retrieved by the
    retrieval system?
    - [x] Ranker
    - [ ] Retriever
    - [ ] Generator
    - [ ] Encoder-decoder

48. Which of these does **NOT** apply when preparing PDF files for OCI Generative AI Agents?
    - [ ] Charts must be two-dimensional with labeled axes.
    - [ ] Reference tables must be formatted with rows and columns.
    - [ ] PDF files can include images and charts.
    - [x] Hyperlinks in PDFs are excluded from chat responses.

49. What does accuracy measure in the context of fine-tuning results for a generative model?
    - [ ] The depth of the neural network layers used in the model
    - [ ] The proportion of incorrect predictions made by the model during an evaluation
    - [x] How many predictions the model made correctly out of all the predictions in an evaluation
    - [ ] The number of predictions a model makes, regardless of whether they are correct or incorrect

50. You are hosting a dedicated AI cluster using the OCI Generative AI service. You need to employ maximum number of
    endpoints due to high workload.

    How many dedicated AI clusters will you require to host at least 60 endpoints?
    - [x] 2
    - [ ] 3
    - [ ] 1
    - [ ] 5