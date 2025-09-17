#!/usr/bin/env python3
"""
MCP服务器环境测试：模拟真实MCP服务器调用环境
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.mcp_nn_tool.mcp_server import train_neural_network_regression
from fastmcp import Context


class MockRequest:
    """模拟MCP请求对象"""
    def __init__(self, user_id=None):
        self.headers = {"user_id": user_id} if user_id else {}


class MockRequestContext:
    """模拟MCP请求上下文"""
    def __init__(self, user_id=None):
        self.request = MockRequest(user_id)


class MockContext:
    """模拟MCP上下文对象"""
    def __init__(self, user_id=None):
        self.request_context = MockRequestContext(user_id)


async def test_mcp_tool_directly():
    """直接测试MCP工具函数（绕过MCP服务器）"""
    test_url = 'http://47.99.180.80/file/uploads/SLM_2.xls'

    print("=== 直接MCP工具测试 ===")
    print(f"URL: {test_url}")

    try:
        # 创建模拟上下文
        mock_ctx = MockContext("test_user_direct")

        # 调用MCP工具函数（但不通过MCP服务器）
        # 注意：这实际上会调用被@mcp.tool()装饰的函数对象
        result = await train_neural_network_regression._func(
            training_file=test_url,
            target_columns=1,
            n_trials=2,
            cv_folds=2,
            num_epochs=3,
            algorithm="TPE",
            loss_function="MAE",
            ctx=mock_ctx
        )

        print("✅ 直接MCP工具调用成功！")
        print(f"任务ID: {result.get('task_id', 'Unknown')}")
        print(f"状态: {result.get('status', 'Unknown')}")
        return True

    except Exception as e:
        print(f"❌ 直接MCP工具调用失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_mcp_tool_with_none_context():
    """测试MCP工具函数（使用None上下文）"""
    test_url = 'http://47.99.180.80/file/uploads/SLM_2.xls'

    print("\n=== MCP工具None上下文测试 ===")
    print(f"URL: {test_url}")

    try:
        # 使用None上下文（模拟某些MCP环境情况）
        result = await train_neural_network_regression._func(
            training_file=test_url,
            target_columns=1,
            n_trials=2,
            cv_folds=2,
            num_epochs=3,
            algorithm="TPE",
            loss_function="MAE",
            ctx=None
        )

        print("✅ None上下文MCP工具调用成功！")
        print(f"任务ID: {result.get('task_id', 'Unknown')}")
        print(f"状态: {result.get('status', 'Unknown')}")
        return True

    except Exception as e:
        print(f"❌ None上下文MCP工具调用失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_network_in_different_contexts():
    """在不同环境下测试网络连接"""
    test_url = 'http://47.99.180.80/file/uploads/SLM_2.xls'

    print("\n=== 网络连接上下文测试 ===")

    contexts = [
        ("直接pandas", lambda: test_direct_pandas(test_url)),
        ("asyncio环境", lambda: test_in_asyncio_context(test_url)),
        ("线程池环境", lambda: test_in_thread_pool(test_url)),
    ]

    results = {}
    for name, test_func in contexts:
        try:
            print(f"\n测试 {name}...")
            success = await test_func()
            results[name] = success
            status = "✅ 成功" if success else "❌ 失败"
            print(f"{name}: {status}")
        except Exception as e:
            print(f"❌ {name} 测试异常: {e}")
            results[name] = False

    return results


async def test_direct_pandas(url):
    """直接测试pandas访问"""
    try:
        import pandas as pd
        data = pd.read_excel(url)
        print(f"直接pandas访问成功，数据形状: {data.shape}")
        return True
    except Exception as e:
        print(f"直接pandas访问失败: {e}")
        return False


async def test_in_asyncio_context(url):
    """在asyncio上下文中测试"""
    try:
        from src.mcp_nn_tool.data_utils import read_data_file
        data = await read_data_file(url)
        print(f"asyncio上下文访问成功，数据形状: {data.shape}")
        return True
    except Exception as e:
        print(f"asyncio上下文访问失败: {e}")
        return False


async def test_in_thread_pool(url):
    """在线程池中测试"""
    try:
        import pandas as pd
        from concurrent.futures import ThreadPoolExecutor

        def load_data():
            return pd.read_excel(url)

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            data = await loop.run_in_executor(executor, load_data)

        print(f"线程池环境访问成功，数据形状: {data.shape}")
        return True
    except Exception as e:
        print(f"线程池环境访问失败: {e}")
        return False


async def main():
    """主测试函数"""
    print("🔍 MCP服务器环境诊断测试")
    print("=" * 50)

    results = {}

    # 测试1: 直接MCP工具调用
    results["mcp_tool_direct"] = await test_mcp_tool_directly()

    # 测试2: None上下文MCP工具调用
    results["mcp_tool_none_ctx"] = await test_mcp_tool_with_none_context()

    # 测试3: 不同网络上下文
    network_results = await test_network_in_different_contexts()
    results.update(network_results)

    # 总结结果
    print("\n" + "=" * 50)
    print("🏁 MCP环境测试结果总结")
    print("=" * 50)

    for test_name, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{test_name:20} : {status}")

    # 分析
    if results.get("mcp_tool_direct") and results.get("mcp_tool_none_ctx"):
        print("\n✅ MCP工具函数本身没问题")
    else:
        print("\n❌ MCP工具函数存在问题")

    if all(network_results.values()):
        print("✅ 网络连接在所有环境下都正常")
    else:
        print("❌ 某些环境下网络连接存在问题")

    print("\n💡 建议:")
    print("1. 检查MCP服务器的网络配置")
    print("2. 检查MCP服务器的SSL/TLS设置")
    print("3. 检查MCP服务器的代理设置")
    print("4. 检查MCP服务器的线程池配置")


if __name__ == "__main__":
    asyncio.run(main())