import os
from typing import Optional

import langchain
from dotenv import load_dotenv
from langchain import PromptTemplate, chains
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from pydantic import ValidationError
from rmrkl import ChatZeroShotAgent, RetryAgentExecutor

from langchain.llms import LlamaCpp, GPT4All
from typing import Optional

from .prompts import FORMAT_INSTRUCTIONS, QUESTION_PROMPT, REPHRASE_TEMPLATE, SUFFIX
from .tools import make_tools


def _make_llm(model, temp, api_key, max_tokens=1000, n_ctx=2048, streaming: bool = False):
    if model.startswith("gpt-3.5-turbo") or model.startswith("gpt-4"):
        try:
            llm = langchain.chat_models.ChatOpenAI(
                temperature=temp,
                model_name=model,
                request_timeout=1000,
                streaming=streaming,
                callbacks=[StreamingStdOutCallbackHandler()],
                openai_api_key=api_key,
            )
        except Exception as e:
            print("Invalid openai key: {}".format(e))
    elif model.startswith("text-"):
        try:
            llm = langchain.OpenAI(
                temperature=temp,
                model_name=model,
                streaming=streaming,
                callbacks=[StreamingStdOutCallbackHandler()],
                openai_api_key=api_key,
            )
        except:
            print("Invalid openai key: {}".format(e))
    elif os.path.exists(model):
        ext = os.path.splitext(model)[-1].lower()
        if ext == ".bin":
            # Assuming this is a GPT4ALL style set of tensors
            llm = GPT4All(model=model, max_tokens=max_tokens, backend='gptj', verbose=False)
        elif ext == ".gguf":
            # Assuming this is a LlamaCpp style set of tensors
            llm = LlamaCpp(
                model_path=model,
                temperature=temp,
                max_tokens=max_tokens,
                n_ctx=n_ctx,
                top_p=1,
                verbose=True, # Verbose is required to pass to the callback manager
                # n_gpu_layers=12
            )
        else:
            raise ValueError(f"Found file: {model}, but this function is only able to load .bin and .gguf models.")    
    else:
        raise ValueError(f"Invalid model name: {model}")
    return llm


class ChemCrow:
    def __init__(
        self,
        tools=None,
        model="gpt-4-0613",
        tools_model="gpt-3.5-turbo-0613",
        temp=0.1,
        max_iterations=40,
        verbose=True,
        streaming: bool = True,
        openai_api_key: Optional[str] = None,
        api_keys: dict = {},
        max_tokens: int = 1000, # Not required for using OpenAI's API
        n_ctx: int =  2048,
    ):
        """Initialize ChemCrow agent."""

        load_dotenv()
        try:
            self.llm = _make_llm(model, temp, openai_api_key, streaming, max_tokens, n_ctx)
            if isinstance(self.llm, str):
                return self.llm
        except ValidationError:
            raise ValueError("Invalid OpenAI API key")

        if tools is None:
            api_keys["OPENAI_API_KEY"] = openai_api_key
            tools_llm = _make_llm(tools_model, temp, openai_api_key, max_tokens, n_ctx, streaming)
            tools = make_tools(tools_llm, api_keys=api_keys, verbose=verbose)

        # Initialize agent
        self.agent_executor = RetryAgentExecutor.from_agent_and_tools(
            tools=tools,
            agent=ChatZeroShotAgent.from_llm_and_tools(
                self.llm,
                tools,
                suffix=SUFFIX,
                format_instructions=FORMAT_INSTRUCTIONS,
                question_prompt=QUESTION_PROMPT,
            ),
            verbose=True,
            max_iterations=max_iterations,
        )

        rephrase = PromptTemplate(
            input_variables=["question", "agent_ans"], template=REPHRASE_TEMPLATE
        )

        self.rephrase_chain = chains.LLMChain(prompt=rephrase, llm=self.llm)

    def run(self, prompt):
        outputs = self.agent_executor({"input": prompt})
        return outputs["output"]
