#!/usr/bin/env python3
"""
Simple MCP Server using FastMCP's built-in streamable-http transport
"""

import argparse
import logging
from pathlib import Path
import sys

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import the MCP server
from src.mcp_nn_tool.mcp_server import mcp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main function to start the MCP server"""
    parser = argparse.ArgumentParser(description="MCP BO Tool Server")
    parser.add_argument('--transport', 
                       default='streamable-http', 
                       choices=['stdio', 'sse', 'streamable-http'],
                       help='Transport type')
    parser.add_argument('--port', 
                       type=int, 
                       default=8090, 
                       help='Port for HTTP-based transports')
    parser.add_argument('--host', 
                       default='localhost', 
                       help='Host for HTTP-based transports')
    
    args = parser.parse_args()
    
    logger.info(f"Starting MCP BO Tool Server...")
    logger.info(f"Transport: {args.transport}")
    
    if args.transport in ['sse', 'streamable-http']:
        logger.info(f"Server will be available at: http://{args.host}:{args.port}")
        logger.info(f"MCP endpoint: http://{args.host}:{args.port}")
        logger.info("Use this URL in your MCP client configuration")
    
    # Run the MCP server with specified transport
    try:
        mcp.run(transport=args.transport, host=args.host, port=args.port)
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        return 1

if __name__ == "__main__":
    exit(main())