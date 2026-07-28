import hashlib
import math
import re

from llm_common import create_chat_completion, create_client


client = create_client()


# -------------------- 1. 准备文档 --------------------

# 先用内存中的几段文档演示 RAG。
# 后续可以把这里替换成从 txt、Markdown 或 PDF 读取内容。
DOCUMENTS = [
    {
        "source": "ThreadLocal 说明",
        "text": (
            "ThreadLocal 是 Java 提供的线程本地变量工具。"
            "每个线程都可以通过同一个 ThreadLocal 对象获取自己独立的变量副本。"
            "ThreadLocal 常用于保存当前用户、数据库连接或请求上下文。"
        ),
    },
    {
        "source": "ThreadLocal 使用注意事项",
        "text": (
            "在线程池场景中使用 ThreadLocal 时，任务结束后应该主动调用 remove。"
            "如果不清理，线程复用时可能读取到上一个任务留下的数据，也可能造成内存泄漏。"
        ),
    },
    {
        "source": "Java 面试基础",
        "text": (
            "Java 中 == 比较基本类型的值，比较引用类型时比较对象地址。"
            "equals 方法通常用于比较对象的内容，但具体行为取决于类是否重写了 equals。"
        ),
    },
]


# -------------------- 2. 文档切分 --------------------


def split_text(text: str, chunk_size: int = 80, overlap: int = 20) -> list[str]:
    """把长文本切成有少量重叠的小文本块。"""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])

        if end >= len(text):
            break

        # 保留 overlap 个字符，让相邻文本块之间保留部分上下文。
        start = end - overlap

    return chunks


def build_chunks(documents: list[dict]) -> list[dict]:
    """切分所有文档，并为每个文本块保存来源。"""
    chunks = []

    for document in documents:
        text_chunks = split_text(document["text"])

        for chunk in text_chunks:
            chunks.append(
                {
                    "source": document["source"],
                    "text": chunk,
                }
            )

    return chunks


# -------------------- 3. 简化版 Embedding --------------------

VECTOR_SIZE = 128


def tokenize(text: str) -> list[str]:
    """提取中文单字、英文单词和数字，作为简化版文本特征。"""
    return re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9]+", text.lower())


def embed_text(text: str) -> list[float]:
    """把文本转换为固定长度的归一化向量。"""
    vector = [0.0] * VECTOR_SIZE

    for token in tokenize(text):
        # 使用稳定哈希把每个 token 映射到向量中的一个位置。
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        index = int(digest, 16) % VECTOR_SIZE
        vector[index] += 1.0

    length = math.sqrt(sum(value * value for value in vector))
    if length == 0:
        return vector

    return [value / length for value in vector]


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """计算两个归一化向量的余弦相似度。"""
    return sum(a * b for a, b in zip(vector_a, vector_b))


# -------------------- 4. 向量检索 --------------------


def build_vector_store(chunks: list[dict]) -> list[dict]:
    """为每个文本块计算向量，形成最简单的向量库。"""
    vector_store = []

    for chunk in chunks:
        vector_store.append(
            {
                **chunk,
                "embedding": embed_text(chunk["text"]),
            }
        )

    return vector_store


def retrieve(
    query: str,
    vector_store: list[dict],
    top_k: int = 2,
) -> list[dict]:
    """根据问题检索最相关的文本块。"""
    query_vector = embed_text(query)
    scored_chunks = []

    for chunk in vector_store:
        score = cosine_similarity(query_vector, chunk["embedding"])
        scored_chunks.append(
            {
                **chunk,
                "score": score,
            }
        )

    scored_chunks.sort(key=lambda item: item["score"], reverse=True)
    return scored_chunks[:top_k]


# -------------------- 5. 让模型基于检索结果回答 --------------------


def answer_with_context(question: str, retrieved_chunks: list[dict]) -> str:
    """把检索到的文本块放进提示词，再请求模型回答。"""
    context = "\n\n".join(
        f"来源：{chunk['source']}\n内容：{chunk['text']}"
        for chunk in retrieved_chunks
    )

    messages = [
        {
            "role": "system",
            "content": "你是一个文档问答助手，只根据参考资料回答问题。资料中没有答案时，请明确说明。",
        },
        {
            "role": "user",
            "content": f"参考资料：\n{context}\n\n问题：{question}",
        },
    ]

    response = create_chat_completion(
        client,
        messages=messages,
        temperature=0.2,
        max_tokens=500,
        extra_body={"thinking": {"type": "disabled"}},
    )

    return response.choices[0].message.content or "模型没有返回内容。"


# 程序启动时只构建一次文本块和向量库。
chunks = build_chunks(DOCUMENTS)
vector_store = build_vector_store(chunks)

print("RAG Agent 已启动，输入问题，输入 exit 退出。")

while True:
    question = input("\n你：").strip()

    if question.lower() == "exit":
        print("对话结束。")
        break

    if not question:
        continue

    retrieved_chunks = retrieve(question, vector_store)

    print("\n检索到的资料：")
    for chunk in retrieved_chunks:
        print(f"- [{chunk['source']}] 相似度：{chunk['score']:.3f}")
        print(f"  {chunk['text']}")

    answer = answer_with_context(question, retrieved_chunks)

    print("\nAgent：")
    print(answer)
