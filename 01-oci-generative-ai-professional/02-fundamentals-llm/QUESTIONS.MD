# Skill Check - Fundamentals of LLMs

1. What does the term "hallucination" refer to in the context of Large Language Models (LLMs)?
    - [ ] A technique used to enhance the model's performance on specific tasks
    - [ ] The process by which the model visualizes and describes images in detail
    - [ ] The model's ability to generate imaginative and creative content
    - [x] The phenomenon where the model generates factually incorrect information or unrelated content as if it were
      true

      **Explanation**: Hallucination occurs when the model generates text that seems plausible or coherent but is not
      grounded in factual reality or relevant to the task at hand. Hallucination can be problematic, especially in
      applications where generating accurate and reliable information is crucial, such as question answering,
      summarization, or content generation for decision-making.

2. What is prompt engineering in the context of Large Language Models (LLMs)?
    - [ ] Adjusting the hyperparameters of the model
    - [ ] Training the model on a large dataset
    - [ ] Adding more layers to the neural network
    - [x] Iteratively refining the ask to elicit a desired response

      **Explanation**: Prompt engineering in the context of Large Language Models (LLMs) refers to the practice of
      designing and refining prompts or input instructions to elicit desired responses from the model. It involves
      crafting specific textual cues or queries that guide the model towards generating outputs that align with the
      user's intentions or requirements.

3. Which statement accurately reflects the differences between these approaches in terms of the number of parameters
   modified and type of data used?
    - [ ] Soft Prompting and Continuous Pretraining are both methods that require no modification to the original
      parameters of the model.
    - [ ] Parameter Efficient Fine-Tuning and Soft Prompting modify all parameters of the model using unlabeled data.
    - [ ] Fine-tuning and Continuous Pretraining both modify all parameters and use labeled, task-specific data.
    - [x] Fine-tuning modifies all parameters using labeled, task-specific data, while Parameter Efficient Fine-Tuning
      updates a few, new parameters also with labeled, task-specific data.

      **Explanation**: Fine-tuning involves adjusting all parameters of the pretrained model using labeled,
      task-specific data. This means that the entire model architecture is modified based on the new task or
      domain-specific data. Parameter Efficient Fine-Tuning updates only a subset of parameters within the pretrained
      model, typically focusing on specific layers or components that are relevant to the new task. Despite this, both
      Fine-tuning and Parameter Efficient Fine-Tuning utilize labeled, task-specific data for training.

4. What does in-context learning in Large Language Models involve?
    - [x] Conditioning the model with task-specific instructions or demonstrations
    - [ ] Pretraining the model on a specific domain
    - [ ] Adding more layers to the model
    - [ ] Training the model using reinforcement learning

      **Explanation**: In-context learning in Large Language Models (LLMs) involves updating or fine-tuning a pretrained
      language model with additional data or examples specific to a particular context or domain. This process enables
      the model to adapt its knowledge and capabilities to better suit the requirements of a specific task or
      application.

5. What is the role of temperature in the decoding process of an LLM?
    - [ ] To increase the accuracy of the most likely word in the vocabulary
    - [x] To adjust the sharpness of the probability distribution over the vocabulary when selecting the next word
    - [ ] To determine the number of words to generate in a single decoding step
    - [ ] To decide which part of speech the next word should belong to

      **Explanation**: When decoding with an LLM, the model assigns probabilities to each word in the vocabulary for the
      next word in the sequence. Temperature controls the sharpness or smoothness of this probability distribution. A
      low temperature value results in a sharper distribution, meaning that the model is more confident in its
      predictions and tends to select the most likely word with higher probability. Conversely, a higher temperature
      value smooths out the distribution, making it more likely for lower probability words to be chosen, leading to
      more diverse and varied output.


