import os
import asyncio
from llama_index.core.workflow import Workflow, step, Event, Context
from llama_index.core.workflow.events import StartEvent, StopEvent, InputRequiredEvent, HumanResponseEvent
from typing import Optional, List, Callable
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings, Document, VectorStoreIndex
from dotenv import load_dotenv
from colorama import Fore, Back, Style
from box_sdk_gen import BoxClient
from llama_index.readers.box import BoxReader
from box_utils.box_client_ccg import get_ccg_user_client
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext
from llama_index.core.extractors import TitleExtractor
from pinecone import Pinecone, FetchResponse

load_dotenv()

Settings.llm = OpenAI(model="gpt-40", temperature=0.1)

class NameStepEvent(InputRequiredEvent):
    """Event to collect the new agents name"""
    pass

class DataStepEvent(InputRequiredEvent):
    """Event to collect the dataset folder ID for the new agent"""
    pass 

class IndexingStepEvent(Event):
    """Event to trigger the indexing step."""
    pass

class CodeGenerationStepEvent(Event):
    """Event to trigger the code generation step."""
    pass

class IntegrationStepEvent(Event):
    """Event to trigger the integration step."""
    pass

class AgentCreationCompleteEvent(Event):
    """Event to signal the completion of agent creation."""
    pass

class ProgressEvent(Event):
    """Event to signal progress to the frontend"""
    pass

class AgentCreatorWorkflow(Workflow):
    """
    Workflow to create a new agent with the following steps:
    1. Collect agent name
    2. Collect dataset folder ID
    """

    @step(pass_context=True)
    async def start(self, ctx: Context, ev: StartEvent) -> InputRequiredEvent:
        """
        Initiates the workflow by asking for the agents name.
        """
        ctx.write_event_to_stream(
            ProgressEvent(msg="Starting Agent Creation")
        )

        return InputRequiredEvent(prefix="Please provide the name of the new agent: ")
    
    @step(pass_context=True)
    async def collect_name(self, ctx: Context, ev: HumanResponseEvent) -> InputRequiredEvent:
        """
        Collects the agents name from the user
        """
        agent_name = str(ev.response.strip())
        if not agent_name:
            ctx.write_event_to_stream(
                ProgressEvent(msg="Agent name cannot be empty. Please provide a valid name.")
            )
            return InputRequiredEvent(prefix="Please provide the name of the new agent:\n")
        
        ctx.write_event_to_stream(
            ProgressEvent(msg=f"Agent name '{agent_name}' received.")
        )
        await ctx.set("agent_name", agent_name)
        return InputRequiredEvent(prefix="Please provide the folder ID where the data for the new agent is stored: ")
    
    @step(pass_context=True)
    async def collect_dataset(self, ctx: Context, ev: HumanResponseEvent) -> IndexingStepEvent:
        """
        Collects the dataset folder ID from the user.
        """
        folder_id = str(ev.response.strip())
        if not folder_id or not folder_id.isdigit():
            ctx.write_event_to_stream(
                ProgressEvent(msg="Folder ID cannot be empty and must only contain digits. Please provide a valid folder ID.")
            )
            return InputRequiredEvent(prefix="Please provide the folder ID where the data for the new agent is stored:")
        
        ctx.write_event_to_stream(
            ProgressEvent(msg=f"Dataset folder ID '{folder_id}' received.")
        )
        await ctx.set("folder_id", folder_id)
        return IndexingStepEvent()
    
    @step(pass_context=True)
    async def indexing_step(self, ctx: Context, ev: IndexingStepEvent) -> CodeGenerationStepEvent:
        """
        Indexes the agent's data using Pinecone and Mistral embeddings.
        """
        folder_id = await ctx.get("folder_id")

        client: BoxClient = get_ccg_user_client(os.getenv("BOX_CLIENT_ID"), os.getenv("BOX_CLIENT_SECRET"), os.getenv("BOX_USER_ID"))

        box_reader = BoxReader(client)

        ctx.write_event_to_stream(
            ProgressEvent(msg=f"Getting documents from box")
        )

        documents: List[Document] = box_reader.load_data(folder_id=folder_id)

        for doc in documents:
            for key in doc.metadata.keys():
                if doc.metadata[key] is None:
                    doc.metadata[key] = ''

        print(len(documents))
        ctx.write_event_to_stream(
            ProgressEvent(msg=f"{len(documents)} documents received from box")
        )

        ctx.write_event_to_stream(
            ProgressEvent(msg=f"Creating Embeddings for documents")
        )

        model_name = "mistral-embed"
        Settings.embed_model = MistralAIEmbedding(model_name=model_name, api_key=os.getenv("MISTRAL_API_KEY"))

        vector_store = PineconeVectorStore(api_key=os.getenv("PINECONE_API_KEY"),index_name=os.getenv("INDEX_NAME"),namespace=folder_id)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

        ctx.write_event_to_stream(
            ProgressEvent(msg=f"Documents Indexed into Pinecone")
        )

        return CodeGenerationStepEvent()
    

    @step(pass_context=True)
    async def generate_code_step(self, ctx: Context, ev: CodeGenerationStepEvent) -> AgentCreationCompleteEvent:
        print("Code Generation Step Completed. Code is in the agents/coffee_agent.py file")
        return AgentCreationCompleteEvent()


    @step(pass_context=True)
    async def completion_step(self, ctx: Context, ev: AgentCreationCompleteEvent) -> StopEvent:
        """
        Completes the workflow and notifies the user.
        """
        agent_name = await ctx.get("agent_name")
        folder_id = await ctx.get("folder_id")
        ctx.write_event_to_stream(
            ProgressEvent(msg=f"Agent '{agent_name}' has been successfully created and integrated with folder id: {folder_id}")
        )
        return StopEvent(result=f"Agent '{agent_name}' creation workflow completed successfully with folder id: {folder_id}")
    
async def main():
    agent_creator = AgentCreatorWorkflow(timeout=1200, verbose=True)

    handler = agent_creator.run()

    async for event in handler.stream_events():
        if isinstance(event, InputRequiredEvent):
            break
    
    response = input(event.prefix)
    handler.ctx.send_event(HumanResponseEvent(response=response))

    async for event in handler.stream_events():
        if isinstance(event, InputRequiredEvent):
            break
    
    response = input(event.prefix)
    handler.ctx.send_event(HumanResponseEvent(response=response))

    async for event in handler.stream_events():
        continue

    final_result = await handler

asyncio.run(main())

