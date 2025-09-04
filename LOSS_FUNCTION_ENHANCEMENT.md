# 损失函数选择功能实现总结

## 概述

本次更新为神经网络训练工具添加了损失函数选择功能，用户现在可以在MAE（Mean Absolute Error）和MSE（Mean Squared Error）之间进行选择。

## 实现的功能

### 1. 训练函数参数扩展

所有相关的训练函数都已添加`loss_function`参数：

- `train_model_cv_fold_with_history()`
- `train_model_cv_fold()`
- `cross_validate_model()`
- `optimize_hyperparameters()`
- `train_final_model()`
- `create_optuna_objective()`

### 2. 损失函数实现

在训练过程中根据`loss_function`参数选择相应的损失函数：

```python
# Select loss function
if loss_function.upper() == "MSE":
    criterion = nn.MSELoss()  # MSE loss
else:  # Default to MAE
    criterion = nn.L1Loss()  # MAE loss
```

### 3. MCP服务器集成

MCP服务器的`train_neural_network`工具现在支持`loss_function`参数：

```python
@mcp.tool()
async def train_neural_network(
    training_file: str,
    target_columns: int = 1,
    n_trials: int = 100,
    cv_folds: int = 5,
    num_epochs: int = 500,
    algorithm: str = "TPE",
    loss_function: str = "MAE",  # 新增参数
) -> Dict[str, Any]:
```

### 4. 学术报告更新

学术报告模板已更新以正确显示所使用的损失函数：

- MAE：显示为 "L1Loss (Mean Absolute Error)"
- MSE：显示为 "MSELoss (Mean Squared Error)"

## 使用方法

### 通过MCP服务器使用

```python
# 使用MAE损失函数
result = await train_neural_network(
    training_file="demo_training_data.csv",
    target_columns=1,
    loss_function="MAE"
)

# 使用MSE损失函数
result = await train_neural_network(
    training_file="demo_training_data.csv",
    target_columns=1,
    loss_function="MSE"
)
```

### 直接调用训练函数

```python
# 在直接调用时指定损失函数
model_states, final_loss, cv_results = await train_final_model(
    best_params=best_params,
    data=data_array,
    feature_number=feature_number,
    target_number=target_number,
    loss_function="MSE"  # 或 "MAE"
)
```

## 测试验证

### 1. 基础功能测试

创建了`simple_loss_test.py`验证：
- MAE和MSE损失函数都能正确应用
- 训练过程正常完成
- 返回合理的损失值

### 2. 集成测试

创建了`test_loss_functions.py`验证：
- 单目标训练支持MAE和MSE
- 多目标训练支持MSE
- 模型保存和文件生成正常

### 3. 报告测试

创建了`test_report_with_loss.py`验证：
- 学术报告正确显示损失函数信息
- MAE显示为"L1Loss (Mean Absolute Error)"
- MSE显示为"MSELoss (Mean Squared Error)"

## 向后兼容性

- 默认损失函数为MAE，保持与之前版本的兼容性
- 所有现有代码无需修改即可继续工作
- 新参数`loss_function`为可选参数

## 文件修改清单

### 核心训练模块
- `src/mcp_nn_tool/training.py`：添加损失函数选择逻辑和报告更新

### MCP服务器
- `src/mcp_nn_tool/mcp_server.py`：添加loss_function参数支持

### 测试文件
- `test_loss_functions.py`：完整的损失函数测试
- `simple_loss_test.py`：基础功能验证
- `test_report_with_loss.py`：报告生成测试

## 测试结果

✅ MAE损失函数：正常工作，报告正确显示
✅ MSE损失函数：正常工作，报告正确显示
✅ 单目标训练：支持MAE和MSE
✅ 多目标训练：支持MSE
✅ 向后兼容性：保持完整兼容
✅ 学术报告：正确显示损失函数信息

## 示例输出

### MAE训练报告片段
```
| Loss Function | L1Loss (Mean Absolute Error) |
```

### MSE训练报告片段
```
| Loss Function | MSELoss (Mean Squared Error) |
```

## 总结

损失函数选择功能已成功实现并通过全面测试。用户现在可以根据具体需求选择适合的损失函数：

- **MAE (L1Loss)**：对异常值更鲁棒，适合存在异常值的数据
- **MSE (L2Loss)**：对较大误差惩罚更重，适合需要平滑预测的场景

该功能完全向后兼容，不会影响现有工作流程。 