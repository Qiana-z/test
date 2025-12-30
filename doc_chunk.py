import re
import json
from collections import defaultdict
from PyPDF2 import PdfReader

# ========== 配置参数 ==========
# 英文报告常见章节标题格式（你可扩展）
CHAPTER_PATTERN = r"^(?:[A-Z][A-Za-z\s\-&,]{3,50})$"

MIN_WORDS = 50  # 过滤掉太短的块

# ========== 辅助函数 ==========
def normalize_text(text: str) -> str:
    """清理多余换行、页眉页脚"""
    text = re.sub(r"Page\s*\d+", "", text)
    text = re.sub(r"Version\s*\d+\.\d+\s*\w*", "", text)
    text = re.sub(r"Getting started with adoption of the TNFD recommendations", "", text)
    return re.sub(r"\n{2,}", "\n", text).strip()

def extract_title_lines(text: str):
    """提取潜在标题行"""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return [l for l in lines if re.match(CHAPTER_PATTERN, l)]

def find_title(line: str):
    """判断是否为章节标题"""
    return re.match(CHAPTER_PATTERN, line)

# ========== 主函数 ==========
def chunk_pdf(pdf_path: str, output_path: str):
    reader = PdfReader(pdf_path)
    pages = [normalize_text(p.extract_text() or "") for p in reader.pages]

    current_title = "Front Matter"
    chapter_map = defaultdict(list)

    for page in pages:
        lines = page.split("\n")
        for line in lines:
            if find_title(line):
                current_title = line.strip()
            else:
                chapter_map[current_title].append(line.strip())

    # 聚合内容
    chunks = []
    for title, lines in chapter_map.items():
        content = " ".join(lines).strip()
        if len(content.split()) < MIN_WORDS:
            continue
        chunks.append({
            "chapter": title,
            "word_count": len(content.split()),
            "content": content
        })

    # 输出 JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"✅ 解析完成，共生成 {len(chunks)} 个章节，已保存到 {output_path}")

# ========== 运行示例 ==========
if __name__ == "__main__":
    input_pdf = "测试/Getting-started-guidance.pdf"   # 你的 TNFD PDF 文件
    output_jsonl = "测试/tnfd_chunks.jsonl"
    chunk_pdf(input_pdf, output_jsonl)

