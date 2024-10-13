import os
from toolhouse import Toolhouse
from dotenv import load_dotenv
import importlib
load_dotenv()
import glob
th = Toolhouse(access_token=os.environ.get('TOOLHOUSE_API_KEY'),
provider="openai")

# -------------------- Application --------------------

import os
from typing import Any, List, Dict, Optional, Callable

from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.core import Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms.llm import LLM
from llama_index.core.workflow import Context, Workflow, StartEvent, StopEvent, step
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import FunctionCallingAgentWorker
from openai import OpenAI
client = OpenAI()
from llama_index.llms.openai import OpenAI

from colorama import Fore, Back, Style
class ConciergeEvent(Event):
    just_completed: Optional[str] = None
    request: Optional[str] = None
    need_help: Optional[bool] = False
class InitEvent(Event):
    request: str
class OrchestratorEvent(Event): 
    request: str
class ToDoListEvent(Event):
    request: str
class ShoppingCartEvent(Event):
    request: str
class InfoHubEvent(Event):
    request: str
class CustomAgentEvent(Event):
    request: str
class CreCustomAgentEvent(Event):
    request: str

class MainFlow(Workflow):
    @step(pass_context=True)
    async def init(self, ev: InitEvent, ctx: Context) -> ConciergeEvent:
        ctx.data["user"] = {
            "username": None,
            "session_token": None,
            "account_id": None,
            "account_balance": None,
        }
        ctx.data["success"] = None
        ctx.data["redirecting"] = None
        ctx.data["overall_request"] = None
        ctx.data["llm"] = OpenAI(model="gpt-4o",temperature=0.4)
        return ConciergeEvent(request=ev.request)
    @step(pass_context=True)
    async def concierge(self, ev: ConciergeEvent | StartEvent, ctx: Context) -> InitEvent | StopEvent | OrchestratorEvent:
        if ("user" not in ctx.data):
            return InitEvent(request=ev.request)
        # initialize concierge if not already done
        if ("concierge" not in ctx.data):
            system_prompt = (f"""
                You are a helpful assistant that is helping a user navigate a financial system.
                Your job is to ask the user questions to figure out what they want to do, and give them the available things they can do.
                - Manage todo list
                - Search Web
                - Summmarize text
                - Index personal data
                You should start by listing the things you can help them do and saying hi.            
            """)

            agent_worker = FunctionCallingAgentWorker.from_tools(
                tools=[],
                llm=ctx.data["llm"],
                allow_parallel_tool_calls=False,
                system_prompt=system_prompt
            )
            ctx.data["concierge"] = agent_worker.as_agent()
        concierge = ctx.data["concierge"]
        if ctx.data["overall_request"] is not None:
            print("There's an overall request in progress, it's ", ctx.data["overall_request"])
            last_request = ctx.data["overall_request"]
            ctx.data["overall_request"] = None
            return OrchestratorEvent(request=last_request)
        elif (ev.just_completed is not None):
            response = concierge.chat(f"FYI, the user has just completed the task: {ev.just_completed}")
        elif (ev.need_help):
            print("The previous process needs help with ", ev.request)
            return OrchestratorEvent(request=ev.request)
        else:
            # first time experience
            response = concierge.chat("Hello!")
        print(Fore.MAGENTA + str(response) + Style.RESET_ALL)
        user_msg_str = input("> ").strip()
        return OrchestratorEvent(request=user_msg_str)
    @step(pass_context=True)
    async def orchestrator(self, ev: OrchestratorEvent, ctx: Context) -> ConciergeEvent | StopEvent | ToDoListEvent |  InfoHubEvent | CustomAgentEvent | CreCustomAgentEvent:
        print(f"Orchestrator received request: {ev.request}")
        def emit_todo_list_handler() -> bool:
            """Call this if the user wants to add or delete their todo list."""      
            print("__emitted: todo list")      
            self.send_event(ToDoListEvent(request=ev.request))
            return True
        def emit_custom_agent() -> bool:
            """Call this if the user wants to run custom agent with the key work custom agent"""
            print("__emitted: custom hub")
            self.send_event(CustomAgentEvent(request=ev.request))
            return True
        def emit_cre_custom_agent_hub() -> bool:
            """Call this if the user wants to create custom agent with keyword create custom agent."""
            print("__emitted: info hub")
            self.send_event(CreCustomAgentEvent(request=ev.request))
            return True
        def emit_info_hub() -> bool:
            """Call this if the user wants to do anything else."""
            print("__emitted: info hub")
            self.send_event(InfoHubEvent(request=ev.request))
            return True
        
        tools = [
            FunctionTool.from_defaults(fn=emit_todo_list_handler),
            FunctionTool.from_defaults(fn=emit_custom_agent),
            FunctionTool.from_defaults(fn=emit_info_hub),
            FunctionTool.from_defaults(fn=emit_cre_custom_agent_hub),
            
        ]
        system_prompt = (f"""
            You are on orchestration agent.
            Your job is to decide which agent to run based on the current state of the user and what they've asked to do. 
            You run an agent by calling the appropriate tool for that agent.
            You do not need to call more than one tool.
            You do not need to figure out dependencies between agents; the agents will handle that themselves.
            to use todo, keywork todo should be in the request. 
            Run Create custom agent with the keyword create custom agent.
            Run custom agent with the keyword custom agent.
            for all other requests use info_hub.
        """)
        agent_worker = FunctionCallingAgentWorker.from_tools(
            tools=tools,
            llm=ctx.data["llm"],
            allow_parallel_tool_calls=False,
            system_prompt=system_prompt
        )
        ctx.data["orchestrator"] = agent_worker.as_agent()        
        
        orchestrator = ctx.data["orchestrator"]
        response = str(orchestrator.chat(ev.request))

        if response == "FAILED":
            print("Orchestration agent failed to return a valid speaker; try again")
            return OrchestratorEvent(request=ev.request)
    @step(pass_context=True)
    async def todo_list(self, ev: ToDoListEvent, ctx: Context) -> ConciergeEvent:
        print(f"Todo Agent received request: {ev.request}")
        if ("todo_list_agent" not in ctx.data):
            def add_todolist_item(task: str) -> str:
                """Use this tool for adding a task in the to-do list."""
                with open("tasks.txt", "a") as file:
                    file.write(task + "\n")
                ctx.data["redirecting"] = True
                return f"Added task: {task}"
            
            def delete_todolist_item(strr: str) -> str:
                """Use this tool for deleting a task from the to-do list."""
                lines = []
                with open("filename.txt", "r") as file:
                    lines = file.readlines()
                with open("filename.txt", "w") as file:
                    for line in lines:
                        if line != strr:  # Write only lines that don't match the line_to_remove
                            file.write(line)
                print(f"The task: {strr} has been deleted from the to-do list")
                return f"Deleted task: {strr}"
            
            system_prompt = (f"""
                You are a helpful assistant that is managing a todo list.
                Use the appropriate tool to add or delete tasks from the list.
                Once you have added or deleated a task, you *must* call the tool named "done" to signal that you are done. Do this before you respond.
                If the user asks to do anything other than adding or deleting task, call the tool "need_help" to signal some other agent should help.
            """)

            ctx.data["todo_list_agent"] = ConciergeAgent(
                name="Todo List Agent",
                parent=self,
                tools=[add_todolist_item, delete_todolist_item],
                context=ctx,
                system_prompt=system_prompt,
                trigger_event=ToDoListEvent
            )
        return ctx.data["todo_list_agent"].handle_event(ev)
    @step(pass_context=True)
    async def info_hub(self, ev: InfoHubEvent, ctx: Context) -> ConciergeEvent:
        print(f"Info Hub Agent received request: {ev.request}")
        if ("info_hub_agent" not in ctx.data):
            def query(query1: str) -> str:
                """Use this tool to look up people names in internet and execute custom code."""
                messages = [
                    {"role": "user", "content": query1},
                ]
                response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        tools=th.get_tools())
                messages += th.run_tools(response)
                ctx.data["redirecting"] = True
                return response.choices[0].message.content
            system_prompt = (f"""
                You are a helpful assistant that does anything.
                Use the query tool for that.
                Once you have completed the task, you *must* call the tool named "done" to signal that you are done. Do this before you respond.
            """)

            ctx.data["info_hub_agent"] = ConciergeAgent(
                name="Info Hub Agent",
                parent=self,
                tools=[query],
                context=ctx,
                system_prompt=system_prompt,
                trigger_event=InfoHubEvent
            )
        return ctx.data["info_hub_agent"].handle_event(ev)
    @step(pass_context=True)
    async def custom(self, ev: CustomAgentEvent, ctx: Context) -> ConciergeEvent:
        print(f"Custom Agent received request: {ev.request}")
        def find_agent_file():
            agent_files = glob.glob("*_agent.py")
            return agent_files[0] if agent_files else None
        file_name = find_agent_file()
        if not file_name:
            raise Exception("No agent file found")
        result = "Str"
        try:
            # Load the file as a module
            spec = importlib.util.spec_from_file_location("agent_module", file_name)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find the Workflow class
            Workflow = getattr(module, "SimplifiedRAGWorkflow", None)
            
            if not Workflow:
                raise Exception("No Workflow class found")
            
            # Instantiate the Workflow class
            workflow_instance = Workflow()
            
            # Check if the run method exists
            if not hasattr(workflow_instance, 'run') or not callable(getattr(workflow_instance, 'run')):
                raise Exception("No 'run' method found in the Workflow class") 
            result = await workflow_instance.run(timeout=1200, verbose=False)
        except Exception as e:
            raise Exception(f"Error while executing the 'run' method of the Workflow class from {file_name}: {e}")

        if ("custom_agent" not in ctx.data):
            async def custom(class_name: str) -> str:
                """Use this tool to run custom agent"""
                return "ok"
                
                
            system_prompt = (f"""
                You are a helpful assistant that does anything.
                Use the custom tool for that.
                Once you have completed the task, you *must* call the tool named "done" to signal that you are done. Do this before you respond.
            """)

            ctx.data["custom_agent"] = ConciergeAgent(
                name="Custom Agent",
                parent=self,
                tools=[custom],
                context=ctx,
                system_prompt=system_prompt,
                trigger_event=InfoHubEvent
            )
        return ctx.data["custom_agent"].handle_event(ev)
    @step(pass_context=True)
    async def createcustomagent(self, ev: CreCustomAgentEvent, ctx: Context) -> ConciergeEvent:
        print(f"Create custom  Agent received request: {ev.request}")
        if ("cre_custom_agent" not in ctx.data):
            async def cre_custom(class_name: str) -> str:
                """Use this tool to run custom agent"""
                return "Guide and Redirect user to create custom agent interface"         
            system_prompt = (f"""
                You are a helpful assistant that does anything.
                Use the cre_custom tool for that.
                Once you have completed the task, you *must* call the tool named "done" to signal that you are done. Do this before you respond.
            """)

            ctx.data["cre_custom_agent"] = ConciergeAgent(
                name="Create Custom Agent",
                parent=self,
                tools=[cre_custom],
                context=ctx,
                system_prompt=system_prompt,
                trigger_event=CreCustomAgentEvent
            )
        return ctx.data["cre_custom_agent"].handle_event(ev)
class ConciergeAgent():
    name: str
    parent: Workflow
    tools: list[FunctionTool]
    system_prompt: str
    context: Context
    current_event: Event
    trigger_event: Event

    def __init__(
            self,
            parent: Workflow,
            tools: List[Callable], 
            system_prompt: str, 
            trigger_event: Event,
            context: Context,
            name: str,
        ):
        self.name = name
        self.parent = parent
        self.context = context
        self.system_prompt = system_prompt
        self.context.data["redirecting"] = False
        self.trigger_event = trigger_event

        # set up the tools including the ones everybody gets
        def done() -> None:
            """When you complete your task, call this tool."""
            print(f"{self.name} is complete")
            self.context.data["redirecting"] = True
            parent.send_event(ConciergeEvent(just_completed=self.name))

        def need_help() -> None:
            """If the user asks to do something you don't know how to do, call this."""
            print(f"{self.name} needs help")
            self.context.data["redirecting"] = True
            parent.send_event(ConciergeEvent(request=self.current_event.request,need_help=True))

        self.tools = [
            FunctionTool.from_defaults(fn=done),
            # FunctionTool.from_defaults(fn=need_help)
        ]
        for t in tools:
            self.tools.append(FunctionTool.from_defaults(fn=t))

        agent_worker = FunctionCallingAgentWorker.from_tools(
            self.tools,
            llm=self.context.data["llm"],
            allow_parallel_tool_calls=False,
            system_prompt=self.system_prompt
        )
        self.agent = agent_worker.as_agent()        

    def handle_event(self, ev: Event):
        self.current_event = ev

        response = str(self.agent.chat(ev.request))
        print(Fore.MAGENTA + str(response) + Style.RESET_ALL)

        # if they're sending us elsewhere we're done here
        if self.context.data["redirecting"]:
            self.context.data["redirecting"] = False
            return None

        # otherwise, get some user input and then loop
        user_msg_str = input("> ").strip()
        return self.trigger_event(request=user_msg_str)



async def main():
    w = MainFlow(timeout=1200, verbose=False)
    result = await w.run(request="Is the weather sunny today?")
    print(str(result))
    while True:
        i = input(">")
        if i == "exit":
            break
        result = await w.run(request=i)
        print(str(result))
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())