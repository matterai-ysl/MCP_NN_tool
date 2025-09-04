#!/usr/bin/env python3
"""Simple script to run the MCP Neural Network Tool server."""

import asyncio
import sys
import os

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def main():
    """Run the FastMCP Neural Network server."""
    try:
        from mcp_nn_tool.mcp_server import mcp
        print("Starting MCP Neural Network Tool server with FastMCP...")
        print("Server will run in STDIO mode for MCP protocol")
        print("Press Ctrl+C to stop the server")
        
        # Run the FastMCP server
        await mcp.run()
        
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 