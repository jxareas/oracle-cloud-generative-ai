<p align="center">
  <img src="images/oci_genai_service.png" width="650" height="300" />
</p>

<div align="center">
    <h1 align = "center">
    <b>OCI Generative AI Service</b>
    </h1>
</div>

This module covers the core capabilities of the Oracle Cloud Infrastructure Generative AI service, such as:

- Using pre-trained foundational models
- Prompt engineering and LLM customization
- Fine-tuning and model inference
- Dedicated AI clusters
- Generative AI security architecture

You can find the [Skill Check Questions here.](./QUESTIONS.md)

## Table of Contents

1. [OCI Generative AI](#oci-generative-ai)
2. [Chat Models](#chat-models)
3. [Embedding Models](#embedding-models)
4. [Prompt Engineering](#prompt-engineering)
5. [Dedicated AI Clusters](#dedicated-ai-clusters)
6. [Fine Tuning](#fine-tuning)
7. [OCI Generative AI Security](#oci-generative-ai-security)

---

## OCI Generative AI

The **OCI Generative AI Service** is a fully managed service that provides a set of customizable Large Language Models (
LLMs) available via a single API to build generative AI applications.

As it is used via a single API, it allows flexibility to use different foundational models with minimal code changes. It
is also of serverless nature, so there is no need to manage any kind of infrastructure.

There are three key characteristics of the service:

- **Choice of models**: Provides high-performing pre-trained foundational models from Meta and Cohere.
- **Flexible fine-tuning**: Create custom models by fine-tuning foundational models with a custom dataset.
- **Dedicated AI clusters**: GPU based compute resources that host fine-tuning and model inference workloads.

### How does OCI Generative AI service work?

![How does the OCI GenAI service work?](images/oci_genai_how_it_works.png)

The OCI Generative AI Service works by following the steps below:

1. Users provide text in natural language via a prompt (e.g: "Describe what is OCI GenAI Service?")
2. The GenAI service processes the input to generate, summarize, transform, extract information or classify text.
3. The response generated is sent back to the user.

### Pre-trained Foundational Models

In OCI GenAI, there are two types of pre-trained foundational models: chat models and embedding models.

![OCI GenAI pre-trained foundational models](./images/oci_genai_pretrained_models.png)

### Fine-tuning

OCI GenAI allows for tine-tuning, which optimizes pre-trained foundational models on smaller domain-specific data to
create customized models. The two main benefits of the fine-tuning process are:

- **_Improved model performance_** on domain-specific tasks.
- **_Increased model efficiency_** by reducing the token usage.

The OCI Generative AI Service enables T-Few fine-tuning a PEFT (_parameter efficient fine-tuning_) which enables fast
and efficient model training, that updates only a fraction of the model's weights.

![OCI GenAI fine-tuning](./images/oci_genai_fine_tuning.png)

### Dedicated AI Clusters

OCI GenAI provides **Dedicated AI Clusters** which are GPU based compute resources that host the customer's fine-tuning
and model inference workloads.

When the GenAI service establishes a dedicated AI cluster, it includes dedicated GPUs and an exclusive RDMA cluster
network for connecting the GPUs.

The GPUs allocated for a given customer's Generative AI tasks are isolated from any other GPUs.

![OCI GenAI AI Clusters](./images/oci_genai_ai_clusters.png)

## Chat Models

<!-- Explanation of conversational AI models and how they are deployed or used within OCI Generative AI. -->

## Embedding Models

<!-- Description of embedding models for semantic understanding, search, and recommendation use cases. -->

## Prompt Engineering

<!-- Techniques and best practices for crafting effective prompts to improve model performance. -->

## Dedicated AI Clusters

<!-- Information about OCI’s infrastructure for running high-performance, isolated AI workloads. -->

## Fine Tuning

<!-- Guide on customizing and refining pretrained models for specific organizational needs. -->

## OCI Generative AI Security

<!-- Overview of OCI’s security measures, compliance, and governance for AI models and data. -->