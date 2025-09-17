#!/usr/bin/env python3
"""
简化调试：创建一个不使用@mcp.tool装饰器的版本来测试
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.mcp_nn_tool.mcp_server import get_user_models_dir, get_user_id, get_task_queue, _train_neural_network_regression_impl
from src.mcp_nn_tool.task_queue import TaskType


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


async def simulate_mcp_train_neural_network_regression(
    training_file: str,
    target_columns: int = 1,
    n_trials: int = 100,
    cv_folds: int = 5,
    num_epochs: int = 500,
    algorithm: str = "TPE",
    loss_function: str = "MAE",
    ctx = None,
):
    """
    模拟MCP工具函数的完整逻辑，但不使用装饰器
    这是train_neural_network_regression函数的完整复制，用于调试
    """
    try:
        # Get user-specific models directory
        user_id = get_user_id(ctx)
        user_models_dir = get_user_models_dir(user_id)

        # Estimate training duration
        estimated_duration = (n_trials * cv_folds * num_epochs * 0.01) + 60

        # Task parameters for tracking
        task_parameters = {
            "training_file": training_file,
            "target_columns": target_columns,
            "n_trials": n_trials,
            "cv_folds": cv_folds,
            "num_epochs": num_epochs,
            "algorithm": algorithm,
            "loss_function": loss_function,
            "task_type": "regression",
            "models_dir": user_models_dir
        }

        # Submit task to queue
        task_queue = get_task_queue(user_models_dir)
        task_id = await task_queue.submit_task(
            task_type=TaskType.REGRESSION_TRAINING,
            task_function=_train_neural_network_regression_impl,
            task_args=(),
            task_kwargs={
                "training_file": training_file,
                "target_columns": target_columns,
                "n_trials": n_trials,
                "cv_folds": cv_folds,
                "num_epochs": num_epochs,
                "algorithm": algorithm,
                "loss_function": loss_function,
                "models_dir": user_models_dir
            },
            task_parameters=task_parameters,
            estimated_duration=estimated_duration
        )

        return {
            "status": "submitted",
            "task_id": task_id,
            "message": "Training task submitted to queue",
            "estimated_duration": estimated_duration,
            "parameters": task_parameters,
            "next_steps": "Use get_training_results(task_id) to check progress and get results when complete"
        }

    except Exception as e:
        print(f"MCP函数模拟失败: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"Failed to submit training task: {str(e)}"
        }


async def test_scenarios():
    """测试不同场景"""
    test_url = 'http://47.99.180.80/file/uploads/SLM_2.xls'

    scenarios = [
        ("正常用户上下文", MockContext("test_user")),
        ("None上下文", None),
        ("空用户ID上下文", MockContext(None)),
        ("空字符串用户ID", MockContext("")),
    ]

    print("🧪 场景测试开始")
    print("=" * 50)

    for scenario_name, ctx in scenarios:
        print(f"\n--- {scenario_name} ---")

        try:
            result = await simulate_mcp_train_neural_network_regression(
                training_file=test_url,
                target_columns=1,
                n_trials=2,
                cv_folds=2,
                num_epochs=3,
                ctx=ctx
            )

            if result["status"] == "submitted":
                print(f"✅ {scenario_name}: 任务提交成功")
                print(f"   任务ID: {result['task_id']}")

                # 等待任务完成
                task_queue = get_task_queue(result["parameters"]["models_dir"])
                for i in range(30):
                    await asyncio.sleep(1)
                    task_status = task_queue.get_task_status(result['task_id'])
                    if task_status and task_status.status.value in ['completed', 'failed']:
                        break

                if task_status:
                    if task_status.status.value == 'completed':
                        print(f"   ✅ 任务完成")
                    else:
                        print(f"   ❌ 任务失败: {task_status.error_message}")

            else:
                print(f"❌ {scenario_name}: 任务提交失败")
                print(f"   错误: {result.get('message', 'Unknown error')}")

        except Exception as e:
            print(f"❌ {scenario_name}: 异常 - {e}")


async def main():
    """主函数"""
    print("🔬 简化MCP调试测试")
    print("这个测试绕过@mcp.tool装饰器，直接测试核心逻辑")

    await test_scenarios()


if __name__ == "__main__":
    asyncio.run(main())