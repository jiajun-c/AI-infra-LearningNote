import numpy as np
from gensim.models import KeyedVectors


def test_static_embedding(model, target_word, related_words1, related_words2, desc1, desc2):
    """
    测试静态向量对多义词的处理能力（基于gensim）
    :param model: gensim加载的KeyedVectors模型
    :param target_word: 多义词
    :param related_words1: 与语义1相关的词汇列表
    :param related_words2: 与语义2相关的词汇列表
    :param desc1: 语义1描述
    :param desc2: 语义2描述
    """
    # 检查目标词是否在模型词表中
    if target_word not in model:
        print(f"⚠️ 词汇'{target_word}'不在模型词表中")
        return
    
    # 1. 验证同一词在不同语境下的向量是否相同
    vec1 = model[target_word]  # 语义1对应的向量（静态模型中唯一）
    vec2 = model[target_word]  # 语义2对应的向量（静态模型中唯一）
    # 计算余弦相似度（静态向量应恒为1.0）
    cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    print(f"\n=== 多义词：'{target_word}' ===")
    print(f"语义1：{desc1}；语义2：{desc2}")
    print(f"两种语义下的向量余弦相似度：{cos_sim:.4f}（=1.0表示向量完全相同）")
    
    # 2. 计算与两种语义相关词汇的相似度
    print(f"\n【与{desc1}相关词汇的相似度】")
    for word in related_words1:
        if word in model:
            # gensim的similarity方法直接计算词向量相似度
            sim = model.similarity(target_word, word)
            print(f"'{target_word}'与'{word}'：{sim:.4f}")
    
    print(f"\n【与{desc2}相关词汇的相似度】")
    for word in related_words2:
        if word in model:
            sim = model.similarity(target_word, word)
            print(f"'{target_word}'与'{word}'：{sim:.4f}")
    
    print("\n结论：静态向量无法区分多义词的不同语义，向量对两种相关词汇均有一定关联")


if __name__ == "__main__":
    # 加载腾讯轻量版中文词向量（需替换为本地文件路径）
    model_path = "./model/lili666/text2vec-word2vec-tencent-chinese/light_Tencent_AILab_ChineseEmbedding.bin"  # 替换为你的模型文件路径
    try:
        # 腾讯模型为二进制格式，binary=True
        model = KeyedVectors.load_word2vec_format(model_path, binary=True)
        print("模型加载成功！")
    except FileNotFoundError:
        from modelscope import snapshot_download
        model_dir = snapshot_download('lili666/text2vec-word2vec-tencent-chinese',cache_dir='./model')
    except Exception as e:
        print(f"模型加载失败：{e}")
        exit()
    
    # 测试："苹果"（水果 vs 品牌）
    test_static_embedding(
        model=model,
        target_word="打",
        related_words_list=[
            ["拳头", "打架", "击打", "殴打"],  # 语义1：击打
            ["游戏", "篮球", "排球", "比赛"],  # 语义2：进行活动
            ["毛衣", "家具", "铁具", "编织"],  # 语义3：制作/编织
            ["电话", "视频", "通讯", "号码"]   # 语义4：通讯
        ],
        desc_list=[
            "击打（如：用拳头打）",
            "进行活动（如：打游戏）",
            "制作/编织（如：打毛衣）",
            "通讯（如：打电话）"
        ]
    )

