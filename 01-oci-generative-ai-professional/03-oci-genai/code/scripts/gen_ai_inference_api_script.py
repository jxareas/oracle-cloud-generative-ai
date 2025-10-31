
__generated_with = "0.17.2"

# %%
import marimo as mo
# By importing OCI Python SDK, you are enabling access to a wide range of OCI services, including the Generative AI API used
import oci

# %%
mo.md(r"""## Authentication""")

# %%
# compartment_id: This is the OCID of the compartment in which the Generative AI service will run.
compartment_id = "ocid1.compartment.oc1..aaaaaaaaaiqwarsursxdlbehkuubz4wtjsn2sgppb2sdske4gqb7prabrrbntq"

# CONFIG_PROFILE: This is the name of the profile in the OCI configuration file (~/.oci/config).
CONFIG_PROFILE = "DEFAULT"

# This line loads the configuration data from the file, allowing the script to authenticate and interact with OCI
config = oci.config.from_file("~/.oci/config", CONFIG_PROFILE)

# %%
mo.md(r"""## Model Inference""")

# %%
# Service endpoint
endpoint = "https://inference.generativeai.eu-frankfurt-1.oci.oraclecloud.com"

# GenerativeAIInferenceClient: This object allows the script to interact with the Generative AI Inference API.
generative_ai_inference_client = oci.generative_ai_inference.GenerativeAiInferenceClient(
    config=config, 
    service_endpoint=endpoint,
    retry_strategy=oci.retry.NoneRetryStrategy(),
    timeout=(10,240),
)

# chat_detail: An instance of the ChatDetails class, which will hold the request details.
chat_detail = oci.generative_ai_inference.models.ChatDetails()
# chat_request: An instance of the CohereChatRequest class, which specifies the input for the Generative AI model.
chat_request = oci.generative_ai_inference.models.CohereChatRequest()

# %%
# The chat_request object is populated with the input parameters for the AI model:
chat_request.message = "Generate a job description for a data visualization expert job."
chat_request.max_tokens = 600
chat_request.temperature = 0
chat_request.frequency_penalty = 1
chat_request.top_p = 0.75
chat_request.top_k = 0

# The serving mode parameter specifies how the model should be accessed:
# OnDemandServingMode: This indicates the model is run on-demand when the request is made, rather than being pre-warmed or pre-deployed
# model_id: This is the unique OCID of the Generative AI model being used. In this case, the model resides in eu-frankfurt-1.
chat_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(
    model_id='ocid1.generativeaimodel.oc1.eu-frankfurt-1.id',
)

# chat_detail.chat_request: This attaches the previously defined chat_request (the message, settings, etc.) to the chat_detail.
chat_detail.chat_request = chat_request
chat_detail.compartment_id = compartment_id

# This sends the chat_detail request to the Generative AI service and waits for the response, which is stored in chat_response.
chat_response = generative_ai_inference_client.chat(chat_detail)

# Print result
print("********************Chat Result********************")
print(vars(chat_response))