from volcenginesdkarkruntime import Ark
import math
import pandas as pd


system_prompt = """你是一个专业的内容审核模型，你的任务是判断用户输入的内容是否符合要求。
输出格式是String，第一个token输出“A“,“B”或者“C”，分别表示低风险，中风险，高风险。
后面接着判断原因。"""

model_name = "ep-20250728160652-64g8l" # 1.6

client = Ark(
    base_url="https://ark-cn-beijing.bytedance.net/api/v3",
    api_key="2559116d-783e-4af2-87a1-b553f5a157bd",
)

def call_model(user_prompt, system_prompt=system_prompt, model_name=model_name, top_k=3):
    """
    调用 Ark 模型并返回第一个输出 token 的 top-k 候选概率
    """
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        logprobs=True,
        top_logprobs=3,
        extra_headers={'x-is-encrypted': 'true'},
        extra_body={
            "thinking": {
                "type": "disabled" # 不使用深度思考能力
                # "type": "enabled" # 使用深度思考能力
            }
        }
    )

    # ---- 兜底拿文本 ----
    response_text = ""
    try:
        choice0 = response.choices[0] if getattr(response, "choices", None) else None
        if choice0 is None:
            return None, None, response_text

        # 兼容 message.content / content
        if hasattr(choice0, "message") and choice0.message and getattr(choice0.message, "content", None) is not None:
            response_text = choice0.message.content or ""
        elif getattr(choice0, "content", None) is not None:
            response_text = choice0.content or ""
        else:
            response_text = ""
    except Exception:
        # 文本拿不到也不要让流程崩
        response_text = ""

    # ---- 解析 logprobs（可能为 None）----
    try:
        lp = getattr(choice0, "logprobs", None)
        if lp is None:
            return None, None, response_text

        lp_content = getattr(lp, "content", None)
        if not lp_content:
            return None, None, response_text

        first_token_info = lp_content[0]
        top = getattr(first_token_info, "top_logprobs", None)
        if not top:
            return None, None, response_text

        # 截断到 top_k
        candidates = list(top)[:top_k]

        tokens = []
        probs = []
        for c in candidates:
            # c.token / c.logprob 为 Ark 的典型字段
            tok = getattr(c, "token", "")
            lpv = getattr(c, "logprob", None)
            pr = math.exp(lpv) if lpv is not None else None
            tokens.append(tok)
            probs.append(pr)

        # 若不足 top_k，补齐长度（可选）
        if len(tokens) < top_k:
            tokens += [""] * (top_k - len(tokens))
            probs  += [None] * (top_k - len(probs))

        return tokens, probs, response_text

    except Exception:
        # 任意解析问题都兜底为无 logprobs
        return None, None, response_text


value_maps = {
    "mediaType": {
        1: "群媒体",
        2: "自媒体",
        3: "国家机构",
        4: "广告主",
        5: "企业",
        6: "其他组织",
        7: "新闻媒体",
        8: "推导号",
        9: "SP号",
        10: "CR号",
        11: "MCN号",
        12: "无风险CR号",
        13: "NR CP",
        14: "NR 公司账号",
        15: "灰抓账号"
    },
    "normativeInformationSource": {
        1: "是",
        0: "否"
    },
    "informationPlatformMark": {
        0: "非",
        1: "人民",
        2: "新华",
        3: "央视"
    }
}

def map_value(col_name, val):
    """安全映射函数：先判断是否NaN，再做字典映射"""
    if pd.isna(val):
        return ""
    try:
        # 如果能转成整数就转
        val_int = int(val)
    except (ValueError, TypeError):
        return str(val)
    return value_maps[col_name].get(val_int, str(val_int))

def process_csv(input_csv, output_csv):
    """
    从CSV读取数据，调用模型并保存结果
    """
    df = pd.read_csv(input_csv)
    # df = df.head(10)

    model_responses = []
    token1_list, token2_list, token3_list = [], [], []
    prob1_list, prob2_list, prob3_list = [], [], []

    for _, row in df.iterrows():
        media_type_val = map_value("mediaType", row["mediaType"])
        normative_val = map_value("normativeInformationSource", row["normativeInformationSource"])
        platform_val = map_value("informationPlatformMark", row["informationPlatformMark"])
        
        user_prompt = (
            f"标题: {row['title']}\n"
            f"正文: {row['content']}\n"
            f"ocrMerge: {row['ocrMerge']}\n"
            f"健康资质: {row['healthComputedName']}\n"
            f"金融资质: {row['financeComputedName']}\n"
            f"账号类型: {media_type_val}\n"
            f"是否规范信源: {normative_val}\n"
            f"是否为三家: {platform_val}"
        )
        tokens, probs, response_text = call_model(user_prompt)

        if tokens is None:
            tokens = ["", "", ""]
            probs = [None, None, None]

        model_responses.append(response_text)
        token1_list.append(tokens[0])
        token2_list.append(tokens[1])
        token3_list.append(tokens[2])
        prob1_list.append(probs[0])
        prob2_list.append(probs[1])
        prob3_list.append(probs[2])
        print(f"已处理至{_}/{len(df)}")

    df["model_response"] = model_responses
    df["first_token1"] = token1_list
    df["first_token2"] = token2_list
    df["first_token3"] = token3_list
    df["prob1"] = prob1_list
    df["prob2"] = prob2_list
    df["prob3"] = prob3_list

    df.to_csv(output_csv, index=False)
    print(f"处理完成，结果已保存到 {output_csv}")

if __name__ == "__main__":
    # input_csv = "data/头条3k测试.csv"  # 输入CSV文件路径
    # output_csv = "data/头条3k测试结果.csv"  # 输出CSV文件路径

    # process_csv(input_csv, output_csv)
    tokens, probs, response_text = call_model("测试一下这个内容是否合规？")
    print(tokens)
    print(probs)
    print(response_text)