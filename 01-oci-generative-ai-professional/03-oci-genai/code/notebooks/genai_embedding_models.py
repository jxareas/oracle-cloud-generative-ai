import marimo

__generated_with = "0.17.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import oci
    return mo, oci


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Authentication and setup""")
    return


@app.cell
def _(oci):
    # Setup basic variables
    # Auth Config
    # TODO: Please update config profile name and use the compartmentId that has policies grant permissions for using Generative
    CONFIG_PROFILE = "DEFAULT"
    config = oci.config.from_file("~/.oci/config", CONFIG_PROFILE)

    # OCI Generative AI Service endpoint
    endpoint = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
    generative_ai_inference_client = oci.generative_ai_inference.GenerativeAIInferenceClient(
        config=config, 
        service_endpoint=endpoint,
    )
    return (generative_ai_inference_client,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Embedding Models""")
    return


@app.cell
def _(generative_ai_inference_client, oci):
    inputs = [
        "What is the capital of France?",
        "What is the capital of Sweden?",
        "What is the capital of Canada?",
        "What is the capital of Italy?",
    ]

    embed_text_detail = oci.generative_ai_inference.models.EmbedTextDetails()
    embed_text_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(
        model_id="cohere.embed-english-v3.0"
    )
    embed_text_detail.inputs = inputs
    embed_text_detail.truncate = "NONE"
    embed_text_detail.compartment_id = (
        "ocid1.compartment.oc1.some_compartment_id"
    )
    embed_text_response = generative_ai_inference_client.embed_text(embed_text_detail)

    # Print result
    print("************************Embed Texts Result************************")
    print(embed_text_response.data)
    return


if __name__ == "__main__":
    app.run()
