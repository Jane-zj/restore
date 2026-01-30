# -*- coding: utf-8 -*-
import os
import json
import base64
import requests

# === 配置 (直接复用您项目中的配置) ===
UPLOAD_API_URL = "https://tt.36588.com.cn/mcard/common/commonUpload"
IMG_URL_PREFIX = "https://tt.36588.com.cn/mcard/assets/resource/imgs/normal/"
REF_DIR = "ref_imgs"  # 您的参考图文件夹名称
FILES = ["1.png", "2.png", "3.png", "4.png"] # 文件名

def upload_file(file_path):
    """读取文件 -> 转Base64 -> 上传 -> 返回URL"""
    if not os.path.exists(file_path):
        print(f"❌ 文件未找到: {file_path}")
        return None
    
    try:
        # 1. 转 Base64
        with open(file_path, "rb") as f:
            b64_str = base64.b64encode(f.read()).decode('utf-8')
        
        # 2. 构造请求
        payload = {"base64Str": f"data:image/jpeg;base64,{b64_str}"}
        headers = {"Content-Type": "application/json"}
        
        # 3. 发送
        print(f"⬆️ 正在上传 {file_path} ...")
        resp = requests.post(UPLOAD_API_URL, json=payload, headers=headers, timeout=60)
        
        if resp.status_code == 200:
            data = resp.json()
            if data.get("success"):
                relative_path = data.get("userData", "")
                full_url = f"{IMG_URL_PREFIX}{relative_path}"
                print(f"✅ 上传成功: {full_url}")
                return full_url
            else:
                print(f"❌ 上传接口报错: {data}")
        else:
            print(f"❌ HTTP 状态码错误: {resp.status_code}")
            
    except Exception as e:
        print(f"❌ 发生异常: {e}")
    
    return None

def main():
    print("🚀 开始批量上传参考图，请稍候...\n")
    
    valid_urls = []
    for filename in FILES:
        path = os.path.join(REF_DIR, filename)
        url = upload_file(path)
        if url:
            valid_urls.append(url)
        else:
            valid_urls.append("UPLOAD_FAILED")

    print("\n" + "="*60)
    print("🎉 获取完成！请直接复制下面的代码替换 CONFIG 中的 REF_IMGS_URLS：")
    print("="*60 + "\n")
    
    print("    # === [⚠️核心修改] 参考图 URL 配置 ===")
    print("    REF_IMGS_URLS = [")
    for u in valid_urls:
        print(f'        "{u}",')
    print("    ]")
    print("\n" + "="*60)

if __name__ == "__main__":
    main()