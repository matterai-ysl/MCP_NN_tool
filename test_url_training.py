#!/usr/bin/env python3
"""
独立测试脚本：测试URL训练功能而不使用MCP装饰器
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.mcp_nn_tool.mcp_server import _train_neural_network_regression_impl
from src.mcp_nn_tool.task_queue import get_task_queue, TaskType, initialize_task_queue


async def test_direct_training():
    """直接测试训练实现函数（不通过MCP工具）"""
    test_url = 'http://47.99.180.80/file/uploads/SLM_2.xls'

    print(f"=== 直接训练测试 ===")
    print(f"URL: {test_url}")

    try:
        result = await _train_neural_network_regression_impl(
            training_file=test_url,
            target_columns=1,
            n_trials=3,
            cv_folds=2,
            num_epochs=5,
            algorithm="TPE",
            loss_function="MAE",
            models_dir="./test_models_direct"
        )

        print("✅ 直接训练成功！")
        print(f"模型ID: {result['model_id']}")
        print(f"最佳MAE: {result['best_mae']}")
        print(f"特征列: {result['feature_names']}")
        print(f"目标列: {result['target_names']}")
        return True

    except Exception as e:
        print(f"❌ 直接训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_task_queue_training():
    """测试通过任务队列的训练（模拟MCP调用）"""
    test_url = 'http://47.99.180.80/file/uploads/SLM_2.xls'

    print(f"\n=== 任务队列训练测试 ===")
    print(f"URL: {test_url}")

    try:
        # 初始化任务队列
        await initialize_task_queue()

        # 获取任务队列
        user_models_dir = "./test_models_queue"
        task_queue = get_task_queue(user_models_dir)

        # 提交任务
        task_id = await task_queue.submit_task(
            task_type=TaskType.REGRESSION_TRAINING,
            task_function=_train_neural_network_regression_impl,
            task_args=(),
            task_kwargs={
                "training_file": test_url,
                "target_columns": 1,
                "n_trials": 3,
                "cv_folds": 2,
                "num_epochs": 5,
                "algorithm": "TPE",
                "loss_function": "MAE",
                "models_dir": user_models_dir
            },
            task_parameters={
                "training_file": test_url,
                "target_columns": 1,
                "task_type": "regression",
                "models_dir": user_models_dir
            },
            estimated_duration=60.0
        )

        print(f"任务已提交，ID: {task_id}")

        # 等待任务完成
        for i in range(60):  # 等待最多60秒
            await asyncio.sleep(1)
            task_status = task_queue.get_task_status(task_id)

            if task_status:
                print(f"任务状态: {task_status.status.value} ({task_status.progress}%)")

                if task_status.status.value == "completed":
                    print("✅ 任务队列训练成功！")
                    if task_status.result:
                        result = task_status.result
                        print(f"模型ID: {result.get('model_id', 'Unknown')}")
                        print(f"最佳MAE: {result.get('best_mae', 'Unknown')}")
                    return True

                elif task_status.status.value == "failed":
                    print(f"❌ 任务队列训练失败: {task_status.error_message}")
                    return False

        print("❌ 任务超时")
        return False

    except Exception as e:
        print(f"❌ 任务队列测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_mcp_function_call():
    """测试MCP函数调用（但不通过MCP服务器）"""
    test_url = 'http://47.99.180.80/file/uploads/SLM_2.xls'

    print(f"\n=== MCP函数调用测试 ===")
    print(f"URL: {test_url}")

    try:
        # 直接调用MCP工具函数的实现，但不通过MCP服务器
        from src.mcp_nn_tool.mcp_server import get_user_models_dir, get_task_queue

        # 模拟用户上下文
        user_id = "test_user"
        user_models_dir = get_user_models_dir(user_id)

        # 模拟MCP函数内部逻辑
        task_queue = get_task_queue(user_models_dir)

        task_parameters = {
            "training_file": test_url,
            "target_columns": 1,
            "n_trials": 3,
            "cv_folds": 2,
            "num_epochs": 5,
            "algorithm": "TPE",
            "loss_function": "MAE",
            "task_type": "regression",
            "models_dir": user_models_dir
        }

        task_id = await task_queue.submit_task(
            task_type=TaskType.REGRESSION_TRAINING,
            task_function=_train_neural_network_regression_impl,
            task_args=(),
            task_kwargs={
                "training_file": test_url,
                "target_columns": 1,
                "n_trials": 3,
                "cv_folds": 2,
                "num_epochs": 5,
                "algorithm": "TPE",
                "loss_function": "MAE",
                "models_dir": user_models_dir
            },
            task_parameters=task_parameters,
            estimated_duration=60.0
        )

        print(f"MCP函数调用任务已提交，ID: {task_id}")

        # 等待完成
        for i in range(60):
            await asyncio.sleep(1)
            task_status = task_queue.get_task_status(task_id)

            if task_status:
                if task_status.status.value == "completed":
                    print("✅ MCP函数调用成功！")
                    return True
                elif task_status.status.value == "failed":
                    print(f"❌ MCP函数调用失败: {task_status.error_message}")
                    return False

        return False

    except Exception as e:
        print(f"❌ MCP函数调用测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """主测试函数"""
    print("🚀 开始URL训练功能独立测试")
    print("=" * 50)

    results = {}

    # 测试1: 直接训练
    results["direct"] = await test_direct_training()

    # 测试2: 任务队列训练
    results["task_queue"] = await test_task_queue_training()

    # 测试3: MCP函数调用
    results["mcp_function"] = await test_mcp_function_call()

    # 总结结果
    print("\n" + "=" * 50)
    print("🏁 测试结果总结")
    print("=" * 50)

    for test_name, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{test_name:15} : {status}")

    # 分析结果
    if all(results.values()):
        print("\n🎉 所有测试都成功！问题可能在MCP服务器环境中。")
    elif results["direct"] and not results["task_queue"]:
        print("\n🔍 直接调用成功，但任务队列失败。问题在任务队列系统。")
    elif results["direct"] and results["task_queue"] and not results["mcp_function"]:
        print("\n🔍 核心功能正常，但MCP函数调用失败。问题在MCP集成层。")
    else:
        print("\n🔍 需要进一步调查问题根源。")


if __name__ == "__main__":
    asyncio.run(main())