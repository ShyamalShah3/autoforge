import os
import asyncio
from dotenv import load_dotenv
from typing import Optional
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core import PromptTemplate
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core.workflow import Workflow, step, Context, StartEvent, StopEvent, Event
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.tools import FunctionTool
from pinecone import Pinecone
from colorama import Fore, Style
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core import get_response_synthesizer

load_dotenv()

# Initialize OpenAI LLM
Settings.llm = OpenAI(model="gpt-4", temperature=0.1)

# Initialize Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
pinecone_index = pc.Index(os.environ["INDEX_NAME"])
vector_store = PineconeVectorStore(pinecone_index=pinecone_index, namespace="288987690990")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
vector_store_index = VectorStoreIndex.from_vector_store(vector_store)

# Define prompt templates
qa_prompt = PromptTemplate(
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)

# Define custom RAG query engine
class RAGStringQueryEngine(CustomQueryEngine):
    """RAG String Query Engine."""
    retriever: BaseRetriever
    response_synthesizer: BaseSynthesizer
    llm: OpenAI
    qa_prompt: PromptTemplate

    def custom_query(self, query_str: str):
        nodes = self.retriever.retrieve(query_str)
        context_str = "\n\n".join([n.node.get_content() for n in nodes])
        response = self.llm.complete(
            self.qa_prompt.format(context_str=context_str, query_str=query_str)
        )
        return str(response)

# Create RAG query engine
rag_query_engine = RAGStringQueryEngine(
    retriever=vector_store_index.as_retriever(),
    response_synthesizer=get_response_synthesizer(response_mode="compact"),
    llm=Settings.llm,
    qa_prompt=qa_prompt,
)

# Create RAG tool
rag_tool = QueryEngineTool.from_defaults(
    rag_query_engine,
    name="rag_tool",
    description="Useful for when you want to learn more about different Coffee Recipes"
)

# Define events
class InterfaceAgentEvent(Event):
    request: Optional[str] = None
    just_completed: Optional[str] = None
    need_help: Optional[bool] = None

class OrchestratorEvent(Event):
    request: str

class RAGEvent(Event):
    request: str

# Define workflow
class SimplifiedRAGWorkflow(Workflow):
    @step(pass_context=True)
    async def interface_agent(self, ctx: Context, ev: InterfaceAgentEvent | StartEvent) -> OrchestratorEvent | StopEvent:
        if "interface_agent" not in ctx.data:
            system_prompt = """
            You are a helpful assistant that helps users navigate information about Coffee Recipees.
            You can help users search for information about different Coffee Recipees.
            Start by explaining what you can help them with.
            """
            agent_worker = FunctionCallingAgentWorker.from_tools(
                tools=[],
                llm=Settings.llm,
                system_prompt=system_prompt
            )
            ctx.data["interface_agent"] = agent_worker.as_agent()

        interface_agent = ctx.data["interface_agent"]
        
        if isinstance(ev, StartEvent):
            response = interface_agent.chat("Hello! I'm an agent that can help you with different coffee recipees. How can I help you today?")
        elif ev.just_completed:
            response = interface_agent.chat(f"I've completed the task: {ev.just_completed}. Is there anything else you'd like to know?")
        else:
            response = interface_agent.chat(ev.request)

        print(Fore.MAGENTA + str(response) + Style.RESET_ALL)
        user_msg = input("> ").strip()
        
        if user_msg.lower() in ["exit", "quit", "stop"]:
            return StopEvent()
        return OrchestratorEvent(request=user_msg)

    @step(pass_context=True)
    async def orchestrator(self, ctx: Context, ev: OrchestratorEvent) -> InterfaceAgentEvent:
        # def use_rag() -> bool:
        #     print("__emitted: RAGEvent")
        #     self.send_event(RAGEvent(request=ev.request))
        #     return True

        def back_to_interface() -> bool:
            print("__emitted: InterfaceAgentEvent")
            self.send_event(InterfaceAgentEvent(request=ev.request))
            return True

        tools = [
            # FunctionTool.from_defaults(fn=use_rag),
            FunctionTool.from_defaults(fn=back_to_interface),
        ]

        system_prompt = """
        You are an orchestration agent. Your job is to decide whether to use the RAG tool to answer the user's question
        about Coffee Recipes, or to send the request back to the interface agent if it's not related
        to that topic. Use the appropriate tool based on the user's request.
        """

        agent_worker = FunctionCallingAgentWorker.from_tools(
            tools=tools,
            llm=Settings.llm,
            system_prompt=system_prompt
        )
        orchestrator = agent_worker.as_agent()
        
        orchestrator.chat(ev.request)

    # @step(pass_context=True)
    # async def rag_agent(self, ctx: Context, ev: RAGEvent) -> InterfaceAgentEvent:
    #     if "rag_agent" not in ctx.data:
    #         system_prompt = """
    #         You are a helpful assistant that provides information about Coffee Recipes.
    #         Use the RAG tool to find relevant information and answer the user's questions.
    #         If the user asks about something unrelated, inform them that you can only help with this specific topic.
    #         Once you have retrieved information about coffee recipes, you must call the tool named "done" to signal that you are done. Do this before you respond.
    #         If the user asks to do anything other than look up information on coffee recipes, call the tool "need_help" to signal some other agent should help.
    #         """
    #         agent_worker = FunctionCallingAgentWorker.from_tools(
    #             tools=[rag_tool],
    #             llm=Settings.llm,
    #             system_prompt=system_prompt
    #         )
    #         ctx.data["rag_agent"] = agent_worker.as_agent()

    #     rag_agent = ctx.data["rag_agent"]
    #     response = rag_agent.chat(ev.request)
        
    #     print(Fore.MAGENTA + str(response) + Style.RESET_ALL)
    #     return InterfaceAgentEvent(just_completed="RAG search")

async def main():
    workflow = SimplifiedRAGWorkflow(timeout=1200, verbose=True)
    result = await workflow.run()
    print(result)

asyncio.run(main())