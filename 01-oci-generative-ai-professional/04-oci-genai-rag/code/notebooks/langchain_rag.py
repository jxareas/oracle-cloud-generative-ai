import marimo

__generated_with = "0.17.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Importing libraries

    In this demo we will explore using LangChain Prompts, Models, Chains and LLMs exposed via a chat API that process sequences of messages as input and output a message.
    """
    )
    return


@app.cell
def _():
    from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
    return (ChatOCIGenAI,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Step 1
    LangChain supports many different language models that you can use interchangeably.
    ChatModels are instances of LangChain Runnables, which means they expose a standard interface
    for interacting with them.
    """
    )
    return


@app.cell
def _(ChatOCIGenAI):
    """Step 1 - LangChain supports many different language models that you can use interchangeably.
    ChatModels are instances of LangChain Runnables, which means they expose a standard interface
    for interacting with them.
    """
    llm = ChatOCIGenAI(
        model_id="cohere.command-r-plus-08-2024",
        service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        compartment_id="ocid1.compartment.oc1.here_you_must_enter_the_compartment_id",
        model_kwargs={'max_tokens': 200}
    )
    return (llm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""After, simply call invoke method on the LLM and pass the input question:""")
    return


@app.cell
def _(llm):
    # Step 2 - Invoke the LLM with a fixed text input
    response = llm.invoke("Tell me one fact about space", temperature=0.7)
    print("Scenario 1 Response -> ")
    print(response.pretty_print())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now, we define a prompt template to structure user inputs. Template allows us to collect inputs at runtime.""")
    return


@app.cell
def _():
    # Step 3 - Use String Prompt to accept text input. Here we create a template and
    # declare an input variable {user_input} and {city}
    from langchain_core.prompts import PromptTemplate

    template = """You are a chatbot having a conversation with a human.
    Human: {user_input} {city}
    !"""

    # Step 4 - here we create a Prompt using the template
    prompt = PromptTemplate(input_variables=["user_input", "city"], template=template)

    # Step 5 - Here we get a prompt value and print it.

    prompt_val = prompt.invoke({"user_input": "Tell us in an exciting tone about", "city": "Las Vegas"})
    print("Prompt String is ->")
    print(prompt_val.to_string())
    return (prompt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Combine the prompt template and the OCI Generative AI model using the pipe operator. In this setup, the invoke method processes the input through the defined chain, producing the desired output.""")
    return


@app.cell
def _(llm, prompt):
    # Step 6 - here we declare a chain that begins with a prompt, next llm
    chain = prompt | llm

    # Step 7 - Next we invoke a chain, get reponse and print it.
    response = chain.invoke({"user_input": "Tell us in an exciting tone about", "city": "New York"})

    print("Scenario 2 Response ->")
    print(response.pretty_print())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Define the chat prompt template.""")
    return


@app.cell
def _():
    # Step 8 - Use Chat Message Prompt to accept text input. Here we create a chat template and
    # use HumanMessage and SystemMessage
    from langchain_core.prompts import ChatPromptTemplate

    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a chatbot that explains in steps."),
            ("human", "{input}"),
        ]
    )

    # Step 9 - Here we get a prompt value and print it.
    prompt_value = chat_prompt.invoke({"input": "What is microbiology?"})
    print(prompt_value)
    return (chat_prompt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Combine the chat prompt template and the OCI Generative AI model using the pipe operator. In this setup, the invoke method processes the input through the defined chain, producing the desired output.""")
    return


@app.cell
def _(chat_prompt, llm):
    # Step 10 - create another chain, get a response and print it.
    chain1 = chat_prompt | llm
    response = chain1.invoke({"input": "What's the New York culture like?"})

    print("Scenario 3 Response ->")
    print(response.pretty_print())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Memory enables LangChain to remember past interactions, improving conversational AI applications.""")
    return


@app.cell
def _(llm):
    # Step 11 - we create a memory and create a chain using the memory.
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationChain

    memory = ConversationBufferMemory()
    conversation = ConversationChain(llm=llm, memory=memory)
    return conversation, memory


@app.cell
def _(conversation, memory):
    # Step 12 - Here we ask our first question and print response and memory contents
    print("\nTurn 1 ---", conversation.invoke("Hello, my name is Hemant!"))
    print("\nMemory Contents", memory.chat_memory)
    return


@app.cell
def _(conversation, memory):
    # Step 13 - Here we ask our followup question and print response and memory contents
    print("\nTurn 2 ---", conversation.invoke("Can you tell what is my name?"))
    print("\nMemory Contents", memory.chat_memory)
    return


if __name__ == "__main__":
    app.run()
