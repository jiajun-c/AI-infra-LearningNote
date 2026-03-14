# 大模型推理能力

传统的推理逻辑是
input -> output

## 1. CoT(chain of thought)

CoT（Chain of thought）是一种用于解决推理问题的方法，其思路是先思考问题，再根据思考结果进行下一步的计算。其推理逻辑是input -> reasoning chain -> output

为了使用他，仅需要在prompt中添加一个 step by step

## 2. TOT(tree of thought)

ToT的推理结果是树状多路径探索，其有回溯能力可以放弃失败路径，计算开销需要多次调用LLM，适用于复杂规划，决策，创造性问题。

## 3. ReAct(Reasoning+Action)

ReAct是一种基于Reasoning+Action的推理方法，其可以统筹整个系统结合外部的工具共同实现最终的目标，在langchain我们可以实现一个ReAct的推理过程

如下所示，先调用agent，然后使用should_continue来判断是否使用工具，然后将tool的结果返回到agent中，再在agent中处理结果，如果输出的结果满足要求，那么结束

```python3
workflow = StateGraph(KatoState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", self.tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "__end__": END})
workflow.add_edge("tools", "agent")
```
