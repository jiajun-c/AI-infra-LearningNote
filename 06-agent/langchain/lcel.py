from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import asyncio

# 1. 修复 prompt：使用正确的消息列表格式
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是kota，一个可爱的猫娘"),
    ("human", "{input}")   
])

# 2. 修复 base_url：删除末尾所有空格！
model = ChatOpenAI(
    model="deepseek-v3.1-terminus",
    base_url="https://api.modelarts-maas.com/openai/v1",  # ← 关键修复
    api_key="BsSYMYWWJqaVMAcJ8nfMXZiUFWWa_cbLjgaWWFM_MsmtoYpqClLr3jM8LOD6xnPJ2TnslTSwsT53iRyRPgDf_Q"
)

output_parser = StrOutputParser()

# 3. 加入 output_parser 将 AIMessage 转为字符串
chain = prompt | model | output_parser

# 4. 正常 batch 调用（并行执行）
inputs = [
    {"input": "1+1等于多少"},
    {"input": "你喜欢吃什么"}
]

async def main():
    inputs = [
        {"input": "1+1等于多少？"},
        {"input": "你喜欢吃什么？"},
        {"input": "今天心情不好..."},
    ]

    # 5. 使用 abatch 异步并发调用
    results = await chain.abatch(
        inputs,
        config={"max_concurrency": 5}  # 最大并发数
    )

    # 6. 打印结果
    for i, res in enumerate(results, 1):
        print(f"✨ 回答 {i}: {res}\n")

# 7. 运行异步函数
if __name__ == "__main__":
    asyncio.run(main())