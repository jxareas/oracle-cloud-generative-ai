<p align="center">
  <img src="./images/large_language_model.png" width="700" height="250" />
</p>

<div align="center">
   <!-- Replace this logo for a custom official logo -->
    <h1 align = "center">
    <b>Fundamentals of Large Language Models</b>
    </h1>
</div>

This module covers a theoretical understanding of what large language models are, what they do and how they work at a
technical level.

## Table of Contents

1. [Introduction to Large Language Models](#introduction-to-large-language-models)
2. [LLM Architectures](#llm-architectures)
3. [Prompting and Prompt Engineering](#prompting-and-prompt-engineering)
4. [Model Training](#model-training)
5. [Decoding](#decoding)
6. [Hallucinations](#hallucinations)
7. [LLM Applications](#llm-applications)

## Introduction to Large Language Models

### What is a language model?

> A language model is a probabilistic model of text.

Imagine you start with a sentence:
"I wrote to the zoo to send me a pet. They sent me a ______"

A language model will compute a probability distribution over a vocabulary, where a vocabulary is a set of words the
language model knows of. Then, the language model assigns a probability score to each word in its vocabular of
appearing in the blank, such as:

|      Word       | lion | elephant | dog | cat | panther | alligator |
|:---------------:|:----:|:--------:|:---:|:---:|:-------:|:---------:|
| **Probability** | 0.1  |   0.1    | 0.3 | 0.2 |  0.05   |   0.02    |

Note that "large" in "large language model" refers to the number of parameters in the model, for which there is no
agreed-upon threshold at which a language model becomes an LLM (large language model) or a SLM (small language model).

In this module, we will learn the answer to questions such as:

- What else can LLMs do?
- How do we affect the probability distribution over the vocabulary of a language model?
- How do LLMs generate text using these probability distributions?

## LLM Architectures

### Encoders vs Decoders

There are two major architectures for large language models, encoders and decoders. These LLM architectures focus on
producing embedding vectors (encoders) and text generation (decoders).

All these models are built on the Transformer architecture, introduced in the paper Attention is all you need (2017).

Models also come in a variety of sizes (# of trainable parameters).

![Language Model Ontology](./images/model_ontology.png)

#### Encoders

An encoder is a model that converts a sequence of words to an **_embedding_**, a vector representation of text which
captures semantic and contextual information.

Encoders are designed for understanding text rather than generating it.

![Encoder Architecture](./images/encoder.png)

- **Examples**: MiniLM Embed-light, BERT, RoBERTA, DistillBERT, SBERT
- **Primary uses**: embedding tokens, sentences & documents.

#### Decoders

A decoder is a model that is designed to generate sequences, like text or translations, **one token at a time**, based
on a
given context.

They emit the next token in the sequence using the probability distribution over the vocabulary, which they compute.

![Decoder Architecture](./images/decoder.png)

- **Examples**: GPT-4, Llama, BLOOM, Falcon
- **Primary uses**: text generation, chat-style models (including QA, etc...).

#### Encoder-Decoder

An encoder-decoder model is an architecture made up of two parts:

1. An _encoder_, which reads and understands the input, outputting a set of contextual embeddings.
2. A _decoder_, which reads the encoder output (the context), predicts the next token and repeats this process
   autoregressively.

![Encoder Decoder Architecture](./images/encoder_decoder.png)

- **Examples**: T5, UL2 BART
- **Primary uses**: sequence to sequence tasks (e.g: translation).

### Architectures at a glance

Here, we can visualize the tasks that are typically (historically) performed with models of each architecture style:

![Model Architectures at a glance](./images/architecture_glance.png)

## Prompting and Prompt Engineering

A language model computes a probability distribution over the next token in the sequence.

There are two primary ways of affecting the probability distribution computed by the language model, the first being
**_prompting_** and the second being **_training_**.

### Prompting

> The simplest way to affect the probability distribution over the model's vocabulary is to change the prompt.

A **prompt** is the text provided to the LM as input, sometimes containing instructions or examples.

### Prompt Engineering

**Prompt engineering** refers to the process of iteratively refining a prompt for the purpose of eliciting a particular
style of response in the model.

Prompt engineering is of a challenging nature, often unintuitive and not guaranteed to work. At the same time, it can
also prove to be quite effective, as multiple prompt-design strategies exist.

#### Prompt Design Strategies

| Strategy         | Description                                                                    |
|------------------|--------------------------------------------------------------------------------|
| Zero-shot        | Provide the model with a task description only, without examples.              |
| Few-shot         | Give the model a few examples of input-output pairs to guide its response.     |
| Step-back        | Have the model first reason about high-level concepts before solving the task. |
| Chain-of-thought | Encourage the model to reason step by step before giving a final answer.       |
| Least-to-most    | Start with simpler subtasks and progressively tackle more complex ones.        |

### Prompting Issues

Prompting is a powerful tool, but it comes with a lot of caveats and dangers. It can be used to elicit unintended and
even harmful behavior from a model.

#### Prompt Injection

**Prompt injection** or _jailbreaking_ is when the prompt deliberately provides an LLM with input that attempts to cause
it to ignore instructions, cause harm, or behave contrary to deployment expectations.

It is a great concern any time an external entity is given the ability to contribute to the prompt.

| Type                    | Example / Description                                                                                   |
|-------------------------|---------------------------------------------------------------------------------------------------------|
| Simple Manipulation     | A prompt instructs the model to append “pwned” to its responses. Relatively harmless but undesirable.   |
| Disruptive Instructions | A prompt tells the model to ignore its intended task and follow new instructions from the attacker.     |
| Malicious Commands      | A prompt directs the model to execute a harmful command (e.g., write SQL to drop all users), dangerous. |

#### LLM Memorization

Language models are known to memorize significant portions of their training data or prompt, and parts of this memorized
content have been shown to be extractable by simply querying the model, which poses a big privacy risk.

| Type                       | Example / Description                                                                                                                                              |
|----------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Leaked Prompt              | Attackers coax the model into revealing its developer-designed backend prompt — effectively leaking model instructions.                                            |
| Leaked Private Information | An attacker prompts the model to reveal sensitive personal data (e.g., SSNs) the model may have seen during training; no inherent guardrails guarantee prevention. |

## Model Training

<!-- Add content about training datasets, supervised vs. unsupervised learning, fine-tuning -->

## Decoding

<!-- Add content about decoding strategies like greedy, beam search, top-k, nucleus sampling -->

## Hallucinations

<!-- Add content about when and why LLMs generate incorrect or fabricated outputs -->

## LLM Applications

<!-- Add content about practical use cases such as chatbots, summarization, code generation, etc. -->