#!/usr/bin/env python3
"""
多模式MCP神经网络服务器启动脚本

支持两种通信模式：
1. stdio模式 - 标准输入输出，适用于传统MCP客户端
2. SSE模式 - Server-Sent Events，适用于CherryStudio等Web客户端

使用方法：
  python run_mcp_server.py --mode stdio    # stdio模式
  python run_mcp_server.py --mode sse      # SSE模式，默认端口8080
  python run_mcp_server.py --mode sse --host 0.0.0.0 --port 9000  # 自定义SSE配置
"""

import sys
import os
import argparse
import asyncio
import uvicorn
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# 导入MCP相关模块
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from mcp.server import Server
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Mount, Route

# 导入我们的MCP服务器
from src.mcp_nn_tool.mcp_server import mcp

def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    """为SSE模式创建Starlette应用."""
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> None:
        """处理SSE连接."""
        # 添加初始化延迟，确保服务器完全准备好
        await asyncio.sleep(0.5)
        
        async with sse.connect_sse(
            request.scope,
            request.receive,
            request._send,
        ) as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options(),
            )

    return Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )

async def run_stdio_mode():
    """运行stdio模式."""
    print("[启动] MCP神经网络服务器 (stdio模式)")
    print("[配置] 模型将保存到: ./trained_model")
    print("[提示] 此模式适用于传统MCP客户端")
    print("="*50)
    
    # 直接运行FastMCP服务器
    await mcp.run()

def run_sse_mode(host: str = "0.0.0.0", port: int = 8080, debug: bool = True):
    """运行SSE模式."""
    print("[启动] MCP神经网络服务器 (SSE模式)")
    print(f"[网络] 服务器地址: http://{host}:{port}")
    print(f"[端点] SSE端点: http://{host}:{port}/sse")
    print("[配置] 模型将保存到: ./trained_model")
    print("[提示] 此模式适用于CherryStudio等Web客户端")
    print("="*50)
    
    # 获取底层MCP服务器并创建Starlette应用
    mcp_server = mcp._mcp_server
    starlette_app = create_starlette_app(mcp_server, debug=debug)
    
    # 使用uvicorn运行服务器
    uvicorn.run(
        starlette_app, 
        host=host, 
        port=port, 
        log_level="debug" if debug else "info"
    )

def main():
    """主函数，解析命令行参数并启动相应模式."""
    parser = argparse.ArgumentParser(
        description="MCP神经网络服务器 - 支持多种通信模式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s --mode stdio                    # stdio模式（传统MCP客户端）
  %(prog)s --mode sse                      # SSE模式（Web客户端，默认8080端口）
  %(prog)s --mode sse --port 9000          # SSE模式，自定义端口
  %(prog)s --mode sse --host 127.0.0.1     # SSE模式，绑定本地地址

支持的工具函数:
  - train_neural_network / train_neural_network_from_content
  - train_classification_model / train_classification_from_content
  - predict_from_file / predict_from_content
  - predict_from_values
  - list_models
  - get_model_info
  - delete_model
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["stdio", "sse"],
        default="stdio",
        help="通信模式：stdio（标准输入输出）或 sse（Server-Sent Events）"
    )
    
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="SSE模式的绑定地址（默认: 0.0.0.0）"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="SSE模式的端口号（默认: 8080）"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式"
    )
    
    args = parser.parse_args()
    
    print(f"[模式] 通信模式: {args.mode.upper()}")
    if args.mode == "sse":
        print(f"[网络] 网络配置: {args.host}:{args.port}")
    if args.debug:
        print("[调试] 调试模式: 启用")
    
    try:
        if args.mode == "stdio":
            # stdio模式
            asyncio.run(run_stdio_mode())
        elif args.mode == "sse":
            # SSE模式（同步函数，因为uvicorn.run是同步的）
            run_sse_mode(host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        print("\n[停止] 服务器已停止")
    except Exception as e:
        print(f"[错误] 服务器启动失败: {e}")
        import traceback
        if args.debug:
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 