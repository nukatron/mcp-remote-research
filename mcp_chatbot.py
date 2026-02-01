import asyncio
import os
from contextlib import AsyncExitStack
from typing import cast

import nest_asyncio
from anthropic import AsyncAnthropic
from anthropic.types import ToolUseBlock
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.sse import sse_client

nest_asyncio.apply()

load_dotenv()

CLAUDE_MODEL = "claude-opus-4-5-20251101"


class MCP_ChatBot:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.anthropic = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.available_tools = []
        self.available_prompts = []
        self.sessions = {}

    async def process_query(self, query: str) -> None:
        """Process a user query through Claude with tool support."""
        messages = [{"role": "user", "content": query}]

        while True:
            response = await self.anthropic.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=2024,
                tools=self.available_tools,
                messages=messages,
            )

            # Add the assistant's response to history
            messages.append({"role": "assistant", "content": response.content})

            # Check if the assistant wants to use tools
            tool_use_blocks = [c for c in response.content if c.type == "tool_use"]

            if not tool_use_blocks:
                # No tools used - print text response and exit
                for content in response.content:
                    if content.type == "text":
                        print(content.text)
                break

            # Process all tool calls in the current response
            tool_results = []
            for block in tool_use_blocks:
                tool_block = cast(ToolUseBlock, block)
                tool_id = tool_block.id
                tool_name = tool_block.name
                tool_args = tool_block.input

                print(
                    f"Calling tool: {tool_name} (ID: {tool_id}) with args: {tool_args}"
                )

                # Call the tool through the MCP session
                result = await self.session.call_tool(
                    name=tool_name, arguments=tool_args
                )

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": str(result),
                    }
                )

            # Add the tool results to history
            messages.append({"role": "user", "content": tool_results})

    async def get_resource(self, resource_uri: str) -> None:
        """Fetch and display a resource."""
        session = self.sessions.get(resource_uri)

        # Fallback for papers:// URIs
        if not session and resource_uri.startswith("papers://"):
            session = next(
                (
                    sess
                    for uri, sess in self.sessions.items()
                    if uri.startswith("papers://")
                ),
                None,
            )

        if not session:
            print(f"Resource '{resource_uri}' not found.")
            return

        try:
            result = await session.read_resource(uri=resource_uri)
            if result and result.contents:
                print(f"\nResource: {resource_uri}")
                print("Content:")
                print(result.contents[0].text)
            else:
                print("No content available.")
        except Exception as e:
            print(f"Error: {e}")

    async def list_prompts(self) -> None:
        """List all available prompts."""
        if not self.available_prompts:
            print("No prompts available.")
            return

        print("\nAvailable prompts:")
        for p in self.available_prompts:
            print(f"- {p['name']}: {p['description']}")
            if p["arguments"]:
                print("  Arguments:")
                for arg in p["arguments"]:
                    arg_name = arg.name if hasattr(arg, "name") else arg.get("name", "")
                    print(f"    - {arg_name}")

    async def execute_prompt(self, prompt_name: str, args: dict) -> None:
        """Execute a prompt with arguments."""
        session = self.sessions.get(prompt_name)
        if not session:
            print(f"Prompt '{prompt_name}' not found.")
            return

        try:
            result = await session.get_prompt(prompt_name, arguments=args)
            if result and result.messages:
                content = result.messages[0].content
                text = (
                    content
                    if isinstance(content, str)
                    else getattr(content, "text", None)
                    or " ".join(getattr(i, "text", str(i)) for i in content)
                )
                print(f"\nExecuting prompt '{prompt_name}'...")
                await self.process_query(text)
        except Exception as e:
            print(f"Error: {e}")

    async def chat_loop(self) -> None:
        """Run an interactive chat loop."""
        print(
            "\n🤖 MCP Chatbot | Commands: quit | @folders | @<topic> | /prompts | /prompt <name> <args>\n"
        )

        while True:
            try:
                query = input("Query: ").strip()
                if not query or query.lower() == "quit":
                    break

                # Handle @resource syntax
                if query.startswith("@"):
                    topic = query[1:]
                    uri = (
                        "papers://folders"
                        if topic == "folders"
                        else f"papers://{topic}"
                    )
                    await self.get_resource(uri)
                    continue

                # Handle /command syntax
                if query.startswith("/"):
                    parts = query.split()
                    command = parts[0].lower()

                    if command == "/prompts":
                        await self.list_prompts()
                    elif command == "/prompt":
                        if len(parts) < 2:
                            print("Usage: /prompt <name> <arg1=value1> <arg2=value2>")
                            continue

                        prompt_name = parts[1]
                        args = {}

                        # Parse arguments
                        for arg in parts[2:]:
                            if "=" in arg:
                                key, value = arg.split("=", 1)
                                args[key] = value

                        await self.execute_prompt(prompt_name, args)
                    else:
                        print(f"Unknown command: {command}")
                    continue

                # Process regular query
                await self.process_query(query)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def connect_to_server(self) -> None:
        """Connect to the MCP server and register tools, prompts, and resources."""
        try:
            remote_research_url = "https://mcp-remote-research-gsu1.onrender.com/sse"
            stdio_transport = await self.exit_stack.enter_async_context(
                sse_client(url=remote_research_url)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            self.session = session

            # Initialize the connection
            await self.session.initialize()

            # Register tools
            for tool in (await session.list_tools()).tools:
                self.sessions[tool.name] = session
                self.available_tools.append(
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema,
                    }
                )

            # Register prompts
            prompts_response = await session.list_prompts()
            if prompts_response and prompts_response.prompts:
                for prompt in prompts_response.prompts:
                    self.sessions[prompt.name] = session
                    self.available_prompts.append(
                        {
                            "name": prompt.name,
                            "description": prompt.description,
                            "arguments": prompt.arguments,
                        }
                    )

            # Register resources
            resources_response = await session.list_resources()
            if resources_response and resources_response.resources:
                for resource in resources_response.resources:
                    self.sessions[str(resource.uri)] = session

        except Exception as e:
            print(f"\nError connecting to server: {str(e)}")

    async def cleanup(self) -> None:
        """Clean up resources."""
        await self.exit_stack.aclose()


async def main() -> None:
    """Main entry point for the chatbot."""
    chatbot = MCP_ChatBot()
    try:
        await chatbot.connect_to_server()
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
