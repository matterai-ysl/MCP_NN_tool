#!/usr/bin/env python3
"""
测试新的MCP URL功能端点
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.mcp_nn_tool.mcp_server import get_user_id, get_user_models_dir
from src.mcp_nn_tool.data_utils import read_data_file


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


async def test_new_endpoint():
    """测试新的URL功能端点"""
    test_url = 'http://47.99.180.80/file/uploads/SLM_2.xls'

    print("🧪 测试新的URL功能端点")
    print("=" * 50)
    print(f"测试URL: {test_url}")

    try:
        # 创建模拟上下文
        mock_ctx = MockContext("test_user_endpoint")

        # 复制test_url_functionality的逻辑
        print("\n执行 URL 功能测试...")

        # Get user context
        user_id = get_user_id(mock_ctx)
        user_models_dir = get_user_models_dir(user_id)

        result = {
            "status": "running",
            "test_url": test_url,
            "user_id": user_id,
            "user_models_dir": user_models_dir,
            "tests": {},
            "environment_info": {}
        }

        # Test 1: Direct URL ping/accessibility
        try:
            import requests
            response = requests.head(test_url, timeout=10)
            result["tests"]["url_accessibility"] = {
                "status": "success",
                "http_status": response.status_code,
                "headers": dict(response.headers),
                "message": f"URL accessible, HTTP {response.status_code}"
            }
        except Exception as e:
            result["tests"]["url_accessibility"] = {
                "status": "failed",
                "error": str(e),
                "message": "URL not accessible via requests.head()"
            }

        # Test 2: Direct pandas read
        try:
            import pandas as pd
            data = pd.read_excel(test_url)
            result["tests"]["direct_pandas"] = {
                "status": "success",
                "data_shape": data.shape,
                "columns": list(data.columns),
                "message": f"Direct pandas.read_excel() successful, shape: {data.shape}"
            }
        except Exception as e:
            result["tests"]["direct_pandas"] = {
                "status": "failed",
                "error": str(e),
                "message": "Direct pandas.read_excel() failed"
            }

        # Test 3: Our custom read_data_file function
        try:
            data = await read_data_file(test_url, max_retries=3, retry_delay=1.0)
            result["tests"]["custom_read_data_file"] = {
                "status": "success",
                "data_shape": data.shape,
                "columns": list(data.columns),
                "message": f"Custom read_data_file() successful, shape: {data.shape}"
            }
        except Exception as e:
            result["tests"]["custom_read_data_file"] = {
                "status": "failed",
                "error": str(e),
                "message": "Custom read_data_file() failed"
            }

        # Test 4: Threading context test
        try:
            import concurrent.futures
            import threading
            import pandas as pd

            def load_in_thread():
                return pd.read_excel(test_url)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(load_in_thread)
                thread_data = future.result(timeout=30)

            result["tests"]["threading_context"] = {
                "status": "success",
                "data_shape": thread_data.shape,
                "thread_id": threading.get_ident(),
                "message": f"Threading context successful, shape: {thread_data.shape}"
            }
        except Exception as e:
            result["tests"]["threading_context"] = {
                "status": "failed",
                "error": str(e),
                "message": "Threading context failed"
            }

        # Environment diagnostics
        result["environment_info"] = {
            "python_version": sys.version,
            "working_directory": os.getcwd(),
            "ssl_context_available": hasattr(__import__('ssl'), 'create_default_context'),
            "current_time": asyncio.get_event_loop().time()
        }

        # Summary
        successful_tests = sum(1 for test in result["tests"].values() if test["status"] == "success")
        total_tests = len(result["tests"])

        result["status"] = "completed"
        result["summary"] = {
            "successful_tests": successful_tests,
            "total_tests": total_tests,
            "success_rate": f"{successful_tests}/{total_tests}",
            "overall_result": "success" if successful_tests == total_tests else "partial_failure"
        }

        print("\n✅ 端点调用成功！")
        print(f"整体状态: {result.get('status', 'Unknown')}")
        print(f"用户ID: {result.get('user_id', 'Unknown')}")
        print(f"用户目录: {result.get('user_models_dir', 'Unknown')}")

        # 显示测试结果
        tests = result.get('tests', {})
        print(f"\n📊 测试结果总览:")
        for test_name, test_result in tests.items():
            status = test_result.get('status', 'unknown')
            message = test_result.get('message', 'No message')
            emoji = "✅" if status == "success" else "❌"
            print(f"  {emoji} {test_name}: {status}")
            print(f"     {message}")
            if status == "failed" and 'error' in test_result:
                print(f"     错误: {test_result['error']}")

        # 显示总结
        summary = result.get('summary', {})
        if summary:
            print(f"\n📈 总结:")
            print(f"  成功测试: {summary.get('successful_tests', 0)}")
            print(f"  总测试数: {summary.get('total_tests', 0)}")
            print(f"  成功率: {summary.get('success_rate', 'Unknown')}")
            print(f"  整体结果: {summary.get('overall_result', 'Unknown')}")

        # 显示环境信息
        env_info = result.get('environment_info', {})
        if env_info:
            print(f"\n🔧 环境信息:")
            print(f"  Python版本: {env_info.get('python_version', 'Unknown')[:50]}...")
            print(f"  工作目录: {env_info.get('working_directory', 'Unknown')}")
            print(f"  Pandas版本: {env_info.get('pandas_version', 'Unknown')}")
            print(f"  SSL支持: {env_info.get('ssl_context_available', 'Unknown')}")

        return True

    except Exception as e:
        print(f"❌ 端点测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_without_context():
    """测试不使用上下文的情况"""
    test_url = 'http://47.99.180.80/file/uploads/SLM_2.xls'

    print(f"\n🧪 测试无上下文情况")
    print("=" * 30)

    try:
        # Get user context with None
        user_id = get_user_id(None)
        user_models_dir = get_user_models_dir(user_id)

        # Simple test
        data = await read_data_file(test_url, max_retries=2, retry_delay=1.0)

        print("✅ 无上下文调用成功！")
        print(f"用户ID: {user_id}")
        print(f"用户目录: {user_models_dir}")
        print(f"数据形状: {data.shape}")

        return True

    except Exception as e:
        print(f"❌ 无上下文测试失败: {e}")
        return False


async def main():
    """主测试函数"""
    print("🔬 MCP URL端点测试")
    print("=" * 50)

    results = {}

    # 测试1: 带上下文的端点测试
    results["with_context"] = await test_new_endpoint()

    # 测试2: 无上下文的端点测试
    results["without_context"] = await test_without_context()

    # 总结
    print("\n" + "=" * 50)
    print("🏁 测试结果总结")
    print("=" * 50)

    for test_name, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{test_name:20} : {status}")

    if all(results.values()):
        print("\n🎉 所有端点测试都成功！")
        print("现在可以在实际MCP环境中使用 test_url_functionality 工具了。")
    else:
        print("\n🔍 某些测试失败，需要进一步调试。")

    print("\n💡 使用方法:")
    print("在MCP客户端中调用: test_url_functionality")
    print("参数: test_url (可选，默认为 SLM_2.xls)")


if __name__ == "__main__":
    asyncio.run(main())