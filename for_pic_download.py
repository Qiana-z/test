import pandas as pd
import requests
import os


def get_extension(content_type):
    if not content_type:
        return ".jpg"
    if "jpeg" in content_type:
        return ".jpg"
    if "png" in content_type:
        return ".png"
    if "gif" in content_type:
        return ".gif"
    return ".jpg"

def download_picurls_from_csv(csv_file: str, output_dir: str): # csv
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_file)
    print(df.columns)
    urls = df["image_url"]

    # 逐条下载
    for i, url in enumerate(urls):
        try:
            print(f"Downloading ({i+1}/{len(urls)})")
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            
            # 根据响应头决定扩展名
            content_type = response.headers.get("Content-Type", "").lower()
            ext = get_extension(content_type)
            filename = f"{i+1}{ext}"
            
            with open(os.path.join(output_dir, filename), "wb") as f:
                f.write(response.content)
        except Exception as e:
            print(f"❌ Failed to download {url}: {e}")

    print("✅ 所有下载完成！")


if __name__ == "__main__":
    csv_file = "pic_data.csv"    # 你的CSV文件名
    output_dir = "downloaded_images"

    download_picurls_from_csv(csv_file, output_dir)





