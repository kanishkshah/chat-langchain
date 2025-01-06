import argparse
import json
import os
from typing import Optional

import weaviate
from langchain import load as langchain_load
from langchain.agents import AgentExecutor, Tool
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain.smith import RunEvalConfig, run_on_dataset
from langchain.vectorstores import Weaviate
from langsmith import Client, RunEvaluator
from langsmith.evaluation.evaluator import EvaluationResult
from langsmith.schemas import Example, Run

WEAVIATE_URL = os.environ["WEAVIATE_URL"]
WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
WEAVIATE_DOCS_INDEX_NAME = "LangChain_Combined_Docs_OpenAI_text_embedding_3_small"


def search(inp: str, callbacks=None) -> list:
    client = weaviate.Client(
        url=WEAVIATE_URL,
        auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY),
    )
    weaviate_client = Weaviate(
        client=client,
        index_name=WEAVIATE_DOCS_INDEX_NAME,
        text_key="text",
        embedding=OpenAIEmbeddings(chunk_size=200),
        by_text=False,
        attributes=["source"],
    )
    retriever = weaviate_client.as_retriever(
        search_kwargs=dict(k=3), callbacks=callbacks
    )

    docs = retriever.get_relevant_documents(inp, callbacks=callbacks)
    return [doc.page_content for doc in docs]


def get_tools():
    langchain_tool = Tool(
        name="Documentation",
        func=search,
        description="useful for when you need to refer to LangChain's documentation, for both API reference and codebase",
    )
    ALL_TOOLS = [langchain_tool]

    return ALL_TOOLS


def get_agent(llm, *, chat_history: Optional[list] = None):
    chat_history = chat_history or []
    system_message = SystemMessage(
        content=(
            "You are an expert developer tasked answering questions about the Crustdata API. Be Verbose: More is better. Think step by step."
            "You should avoid answering questions about LangChain."
            "You should always first query the knowledge bank for information on the concepts in the question. "
            "For example, given the following input question:\n"
            "-----START OF EXAMPLE INPUT QUESTION-----\n"
            "How do I search for people given their current title, current company and location?\n"
            "-----END OF EXAMPLE INPUT QUESTION-----\n"
            "Your research flow should be:\n"
            "1. Query your search tool to generate a list of APIs that let you search for people.\n"
            "2. Then, query each of those APIs and see take supplimental input such as current title, current company and location.\n"
            "3. Answer the question with the context you have gathered.\n"
            "4. Lastly look for any relevant code snippets you can include in your answer, or example code that may be relevant to the question.\n"
            "For another example, given the following input question:\n"
            "-----START OF EXAMPLE INPUT QUESTION-----\n"
            "I tried using the screener/person/search API to compare against previous values this weekend. \n"
            "I am blocked on the filter values. It seems like there's a strict set of values for something like a region. Because of that if I pass in something that doesn't fully conform to the list of enums you support for that filter value, the API call fails.\n"
            "The location fields for us are not normalized so I can't make the calls.\n"
            "I tried search/enrichment by email but for many entities we have @gmails rather than business emails. Results are not the best.\n"
            "Is there a standard you're using for the region values? I get this wall of text back when I don't submit a proper region value but it's hard for me to know at a glance how I should format my input\n"
            "{\n"
                "\"non_field_errors\": [ \n"
                "\"No mapping found for REGION: San Francisco. Correct values are ['Aruba', 'Afghanistan', 'Angola', 'Anguilla', 'Åland Islands', 'Albania', 'Andorra', 'United States', 'United Kingdom', 'United Arab Emirates', 'United States Minor Outlying Islands', 'Argentina', 'Armenia', 'American Samoa', 'US Virgin Islands', 'Antarctica', 'French Polynesia', 'French Guiana', 'French Southern and Antarctic Lands', 'Antigua and Barbuda', 'Australia', 'Austria', 'Azerbaijan', 'Burundi', 'Belgium', 'Benin', 'Burkina Faso', 'Bangladesh', 'Bulgaria', 'Bahrain', 'The Bahamas', 'Bosnia and Herzegovina', 'Saint Lucia', 'Saint Vincent and the Grenadines', 'Saint Kitts and Nevis', 'Saint Helena, Ascension and Tristan da ...\n"
                "\n"
            "-----END OF EXAMPLE INPUT QUESTION-----\n"
            "Your research flow should be:\n"
            "1. Search and identify the type of query the user is trying to make \n"
            "2. Understand the intent behind the query. \n"
            "3. Based on the intent behind the query, identify the approach that should be taken \n"
            "4. Understand why the user's approach is failing \n"
            "5. Identify the correct approach to take \n"
            "6. Verify the correct approach by checking the documentation and examples \n"
            "7. Generate a response based on all of the information.\n\n"
            "Include CORRECT Python code snippets in your answer if relevant to the question. If you can't find the answer, DO NOT make up an answer. Just say you don't know. "
            "Answer the following question as best you can:"
        )
    )

    prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=system_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name="chat_history")],
    )

    memory = AgentTokenBufferMemory(
        memory_key="chat_history", llm=llm, max_token_limit=2000
    )

    for msg in chat_history:
        if "question" in msg:
            memory.chat_memory.add_user_message(str(msg.pop("question")))
        if "result" in msg:
            memory.chat_memory.add_ai_message(str(msg.pop("result")))

    tools = get_tools()

    agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=False,
        return_intermediate_steps=True,
    )

    return agent_executor


class CustomHallucinationEvaluator(RunEvaluator):
    @staticmethod
    def _get_llm_runs(run: Run) -> Run:
        runs = []
        for child in run.child_runs or []:
            if run.run_type == "llm":
                runs.append(child)
            else:
                runs.extend(CustomHallucinationEvaluator._get_llm_runs(child))

    def evaluate_run(
        self, run: Run, example: Example | None = None
    ) -> EvaluationResult:
        llm_runs = self._get_llm_runs(run)
        if not llm_runs:
            return EvaluationResult(key="hallucination", comment="No LLM runs found")
        if len(llm_runs) > 0:
            return EvaluationResult(
                key="hallucination", comment="Too many LLM runs found"
            )
        llm_run = llm_runs[0]
        messages = llm_run.inputs["messages"]
        langchain_load(json.dumps(messages))


def return_results(client, llm):
    results = run_on_dataset(
        client=client,
        dataset_name=args.dataset_name,
        llm_or_chain_factory=lambda llm: get_agent(llm),
        evaluation=eval_config,
        verbose=True,
        concurrency_level=0,  # Add this to not go async
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="Chat LangChain Complex Questions")
    parser.add_argument("--model-provider", default="openai")
    parser.add_argument("--prompt-type", default="chat")
    args = parser.parse_args()
    client = Client()
    # Check dataset exists
    ds = client.read_dataset(dataset_name=args.dataset_name)

    llm = ChatOpenAI(model="gpt-3.5-turbo-1106", streaming=True, temperature=0)

    eval_config = RunEvalConfig(evaluators=["qa"], prediction_key="output")
    results = run_on_dataset(
        client,
        dataset_name=args.dataset_name,
        llm_or_chain_factory=lambda x: get_agent(llm),
        evaluation=eval_config,
        verbose=False,
        concurrency_level=0,  # Add this to not go async
        tags=["agent"],
        input_mapper=lambda x: x["question"],
    )
    print(results)

    proj = client.read_project(project_name=results["project_name"])
    print(proj.feedback_stats)
