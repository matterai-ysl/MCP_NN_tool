# MCP神经网络服务器使用说明

## 概述

这个MCP服务器支持两种通信模式，可以根据你的客户端类型选择合适的启动方式。

## 支持的通信模式

### 1. stdio模式（推荐用于传统MCP客户端）
- **适用场景**：Claude Desktop、其他传统MCP客户端
- **通信方式**：标准输入输出
- **特点**：直接、高效、稳定

### 2. SSE模式（推荐用于Web客户端）  
- **适用场景**：CherryStudio、Web应用
- **通信方式**：HTTP + Server-Sent Events
- **特点**：支持文件内容直接传输，无需本地文件路径

## 启动命令

### stdio模式
```bash
# 默认启动（stdio模式）
python run_mcp_server.py

# 显式指定stdio模式
python run_mcp_server.py --mode stdio
```

### SSE模式
```bash
# 默认SSE配置（绑定0.0.0.0:8080）
python run_mcp_server.py --mode sse

# 自定义端口
python run_mcp_server.py --mode sse --port 9000

# 自定义主机和端口
python run_mcp_server.py --mode sse --host 127.0.0.1 --port 9000

# 启用调试模式
python run_mcp_server.py --mode sse --debug
```

## 命令行参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--mode` | 选择 | `stdio` | 通信模式：`stdio` 或 `sse` |
| `--host` | 字符串 | `0.0.0.0` | SSE模式的绑定地址 |
| `--port` | 整数 | `8080` | SSE模式的端口号 |
| `--debug` | 布尔 | `False` | 启用调试模式 |

## 客户端配置

### Claude Desktop配置（stdio模式）
```json
{
  "mcpServers": {
    "neural-network": {
      "command": "python",
      "args": ["D:/Code2/MCP-NN-Tool/run_mcp_server.py", "--mode", "stdio"],
      "cwd": "D:/Code2/MCP-NN-Tool"
    }
  }
}
```

### CherryStudio配置（SSE模式）
1. 启动服务器：`python run_mcp_server.py --mode sse`
2. 在CherryStudio中添加MCP服务器：
   - 服务器地址：`http://localhost:8080/sse`
   - 传输方式：SSE

## 支持的工具函数

### 文件输入版本（适用于stdio模式）
- `train_neural_network` - 回归模型训练
- `train_classification_model` - 分类模型训练  
- `predict_from_file` - 从文件批量预测

### 内容输入版本（适用于SSE模式）
- `train_neural_network_from_content` - 从CSV内容训练回归模型
- `train_classification_from_content` - 从CSV内容训练分类模型
- `predict_from_content` - 从CSV内容批量预测

### 通用函数（两种模式都支持）
- `predict_from_values` - 单个/多个数值预测
- `list_models` - 列出所有模型
- `get_model_info` - 获取模型信息
- `delete_model` - 删除模型

## 使用示例

### stdio模式示例
```python
# 训练回归模型
await train_neural_network(
    training_file="/path/to/data.csv",
    target_columns=1,
    n_trials=50,
    cv_folds=5,
    num_epochs=300
)

# 从文件预测
await predict_from_file(
    model_id="your-model-id",
    prediction_file="/path/to/test.csv"
)
```

### SSE模式示例
```python
# 训练分类模型（使用CSV内容）
await train_classification_from_content(
    csv_content="feature1,feature2,label\n1,2,A\n3,4,B",
    n_trials=20,
    cv_folds=3
)

# 从内容预测
await predict_from_content(
    model_id="your-model-id", 
    csv_content="feature1,feature2\n5,6\n7,8"
)
```

## 常见问题

### Q: 如何选择使用哪种模式？
A: 
- 如果使用Claude Desktop等传统MCP客户端，选择stdio模式
- 如果使用CherryStudio等Web客户端，选择SSE模式

### Q: SSE模式启动时出现初始化错误？
A: 这是正常现象，服务器会添加初始化延迟确保稳定性，不影响正常使用。

### Q: 如何在不同端口启动多个服务器实例？
A: 使用不同端口号启动：
```bash
python run_mcp_server.py --mode sse --port 8080  # 实例1
python run_mcp_server.py --mode sse --port 8081  # 实例2
```

### Q: 模型文件保存在哪里？
A: 默认保存在 `./trained_model` 目录下，每个模型有独立的UUID文件夹。

## 技术特性

- ✅ 支持回归和分类任务
- ✅ 自动超参数优化（TPE算法）
- ✅ 交叉验证评估
- ✅ 详细的实验报告
- ✅ 模型持久化存储
- ✅ 多种文件格式支持（CSV、Excel）
- ✅ 异步处理
- ✅ 参数日志记录 