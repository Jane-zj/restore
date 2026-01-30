# -*- coding: utf-8 -*-
"""
@File       : api_server_v14_speed_no_qa_upload.py
@Description: 智能名片翻新 (极速并发 + 无质检 + 自动上传版)
@Logic      : 
    1. 目标：单图输入 -> 并发调用 4 种 Prompt -> 产出 4 张裁剪后的结果。
    2. 变更：图片生成后自动上传至文件服务器。
    3. 兼容：返回的字段名仍为 xxx_base64，但内容实际为 URL (e.g., "https://...")。
@Usage      : uvicorn api_server_v14_speed_no_qa:app --host 0.0.0.0 --port 6006
"""

import os
import json
import base64
import shutil
import threading
import tempfile
import uuid
import requests
import cv2
import time
import numpy as np
from typing import List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from enum import Enum
from dataclasses import dataclass

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from volcenginesdkarkruntime import Ark

# === 本地矫正模块 ===

from test import processor as img_processor

CORRECTION_LOCK = threading.Lock()

# === 全局配置 ===
class CONFIG:
    VOLC_API_KEY = ""
    VOLC_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
    
    MODEL_GEN = "doubao-seedream-4-5-251128"
    MODEL_VISION = "doubao-seed-1-6-vision-250815"

    FIXED_GEN_SIZE = "3000x1824"
    
    # === 并发设置 ===
    MAX_WORKERS = 16 

    # === 测试上传接口配置 ===
    UPLOAD_API_URL = "https://tt.36588.com.cn/acard/common/commonUpload"
    IMG_URL_PREFIX = "https://tt.36588.com.cn/acard/assets/resource/imgs/normal/"
    
    REF_IMG_PATHS = [
        "ref_imgs/1.png", "ref_imgs/2.png", "ref_imgs/3.png", "ref_imgs/4.png"
    ]

    PROMPT_DESCRIBE = """
    平面设计还原专家：透过图像分析原始数字布局，按【背景 / 填充 / 色块 / 排版 / 图标 / 干扰 / 风格】7 部分输出。
    注意：这是对一张已经过矫正的平面图进行分析。
    精准对齐原图布局，去物理化（无反光 / 光影 / 纹理），不读文字，保留设计内实物图（勿误判产品图为干扰）。
    """
    PROMPT_Gen_BASE = """
    严格遵循DESCRIBE的布局分析，将参考图转**标准直角矩形矢量高清设计稿**： 
    * **画布 (Canvas)** = **名片纸张表面 (Card Surface)** 
    - 核心：1:1 还原原图布局，将参考图转标准直角矩形、矢量高清、无噪点、可复用设计稿。消除所有 “磨砂感 / 颗粒感 / 纸张纹理 / 膜面反光”。色块 / 文字 / 图标位置、大小、边界与原图完全一致，
    矢量图，正视图，绝对扁平，无厚度 / 遮挡 / 折痕 / 阴影 / 透视 / 扭曲，绝对矩形，清晰排版，AI 设计稿，去材质化，高保真。
    """
    PROMPT_WithoutVison = """
    名片，干净背景，矢量图，正视图，平面设计原稿，绝对扁平，无厚度，无遮挡，无折痕，无阴影，无透视，无扭曲，绝对矩形，清晰的文字排版，Adobe Illustrator设计稿，去材质化，高保真。
    """
    PROMPT_V2SIMPLE = """
    图 1—图 4为参考图，对图五进行处理：名片，平面设计原稿，矢量图，正视图，图片高清无噪点，扫描仪效果，绝对扁平，无厚度，无遮挡，无折痕，无阴影，无笔迹，无透视，无扭曲，绝对矩形，清晰的文字排版，Adobe Illustrator设计稿，去材质化，高保真，相框，红色边框内。
    """
    PROMPT_V2STRICT = """
    图 1—图 4 仅作为【画质风格参考】（代表高清、矢量、平整、无噪点的**风格**）。
    图 5 是【唯一内容源】（代表必须保留的Logo、文字、排版）。

    请严格执行以下指令对 图5 进行重绘：
    1. **内容忠实度**：必须**100% 锁定**图 5 的原始设计元素。**绝对禁止**从图 1—图 4 中提取任何 Logo、文字或特定的背景图案应用到结果中。
    2. **画质提升**：利用参考图的高清质感，将图 5 的模糊像素转化为清晰的矢量线条。
    
    总结：名片，平面设计原稿，矢量图，正视图，图片高清无噪点，扫描仪效果，绝对扁平，无厚度，无遮挡，无折痕，无阴影，无笔迹，无透视，无扭曲，绝对矩形，清晰的文字排版，Adobe Illustrator设计稿，去材质化，高保真，相框，红色边框内。
    """

    PROMPT_BG_CHECK = """
    色彩分析师：请判断这张图片的**背景设计**是否为【纯色/单色】背景。
    
    判断标准：
    1. 如果背景是单一颜色（允许极轻微的纸张纹理，但整体是单色的），视为 True。
    2. 如果背景有渐变、复杂图案、照片、多色块拼接，视为 False。
    
    请输出纯 JSON 格式，不要包含 Markdown 标记：
    {"is_solid": true/false, "hex_color": "#RRGGBB"}
    
    如果是纯色，请提取最主要的背景 HEX 颜色代码（例如 #FFFFFF 或 #000000）。
    如果不是纯色，hex_color 请返回 null 或 ""。
    """

client = Ark(api_key=CONFIG.VOLC_API_KEY, base_url=CONFIG.VOLC_BASE_URL)
app = FastAPI(title="Smart Card Restore V14 Upload", description="并发4路极速+自动上传返回URL")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# === 工具函数 ===
def _file_to_base64_str(file_path):
    """仅用于构造上传接口的请求体"""
    if not os.path.exists(file_path): return ""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def upload_to_cdn(file_path: str) -> str:
    """
    上传图片到服务器，返回完整URL
    """
    if not os.path.exists(file_path):
        return ""
    
    try:
        # 1. 转Base64用于上传
        raw_b64 = _file_to_base64_str(file_path)
        if not raw_b64: return ""

        b64_with_header = f"data:image/jpeg;base64,{raw_b64}"

        # 2. 构造请求
        payload = {"base64Str": b64_with_header}
        headers = {"Content-Type": "application/json"}
        
        # 3. 发送请求 (设置超时)
        resp = requests.post(CONFIG.UPLOAD_API_URL, json=payload, headers=headers, timeout=120)
        
        # 4. 解析结果
        if resp.status_code == 200:
            data = resp.json()
            if data.get("success"):
                user_data = data.get("userData", "")
                # 拼接完整 URL
                return f"{CONFIG.IMG_URL_PREFIX}{user_data}"
            else:
                print(f"⚠️ Upload Failed: {data}")
        else:
            print(f"⚠️ Upload HTTP Error: {resp.status_code}")
            
    except Exception as e:
        print(f"❌ Upload Exception: {e}")
    
    return ""

def perform_correction_safe(input_path, output_path):
    if img_processor is None: return False
    with CORRECTION_LOCK:
        try:
            res = img_processor.process_image(input_path, "resnet", output_path)
            if not isinstance(res, str) and res is not None:
                 res.save(output_path, quality=95)
            return True
        except Exception as e:
            print(f"❌ ResNet Correct Error: {e}")
            return False

# === 裁剪算法 ===
def _crop_resnet_post(image_path, output_path):
    return perform_correction_safe(image_path, output_path)

def _crop_red_frame(image_path, output_path):
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
        return rect
    try:
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None: return False
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_or(cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255])),
                              cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255])))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return False
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < 2000: return False
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4:
            pts = order_points(approx.reshape(4, 2))
            dst = np.array([[0, 0], [3000-1, 0], [3000-1, 1824-1], [0, 1824-1]], dtype="float32") 
            warped = cv2.warpPerspective(img, cv2.getPerspectiveTransform(pts, dst), (3000, 1824))
            cv2.imencode('.jpg', warped)[1].tofile(output_path)
            return True
        return False
    except: return False

# === 策略定义 ===
class StrategyType(Enum):
    V7_STATIC = "v7_static"
    V7_DYNAMIC = "v7_dynamic"
    V8_STRICT = "v8_strict"
    V8_SIMPLE = "v8_simple"

@dataclass
class RestoreStrategy:
    name: str
    type: StrategyType
    use_layout_analysis: bool
    use_ref_images: bool
    crop_func: Callable

    def get_prompt(self, layout_desc: str = "") -> str:
        if self.type == StrategyType.V7_DYNAMIC:
            return f"{CONFIG.PROMPT_Gen_BASE}\n视觉参考：{layout_desc}"
        elif self.type == StrategyType.V7_STATIC:
            return CONFIG.PROMPT_WithoutVison
        elif self.type == StrategyType.V8_SIMPLE:
            return CONFIG.PROMPT_V2SIMPLE
        elif self.type == StrategyType.V8_STRICT:
            return CONFIG.PROMPT_V2STRICT
        return ""

PARALLEL_STRATEGIES = [
    RestoreStrategy(name="静态生成", type=StrategyType.V7_STATIC, use_layout_analysis=False, use_ref_images=False, crop_func=_crop_resnet_post),
    RestoreStrategy(name="视觉分析", type=StrategyType.V7_DYNAMIC, use_layout_analysis=True, use_ref_images=False, crop_func=_crop_resnet_post),
    RestoreStrategy(name="内容锁定", type=StrategyType.V8_STRICT, use_layout_analysis=False, use_ref_images=True, crop_func=_crop_red_frame),
    RestoreStrategy(name="参考图", type=StrategyType.V8_SIMPLE, use_layout_analysis=False, use_ref_images=True, crop_func=_crop_red_frame)
]

# === 数据模型 (保持字段名不变，内容改为URL) ===
class GenerationResult(BaseModel):
    strategy_name: str
    crop_image_base64: str  # ⚠️ 实际返回 URL
    gen_image_base64: str   # ⚠️ 实际返回 URL

class BackgroundInfo(BaseModel):
    is_solid: bool; hex_color: Optional[str] = ""

class SingleInputResult(BaseModel):
    filename: str
    status: str
    error_msg: str = ""
    original_image_base64: str = ""    # ⚠️ 实际返回 URL
    corrected_image_base64: str = ""   # ⚠️ 实际返回 URL
    background_info: Optional[BackgroundInfo] = None
    generations: List[GenerationResult] = []

class BatchRestoreResponse(BaseModel):
    total_requested: int; total_success: int; batch_results: List[SingleInputResult]
class UrlBatchRequest(BaseModel):
    urls: List[str]

# === 基础功能 ===
def _extract_json(c):
    try: return json.loads(c.split("```json")[1].split("```")[0] if "```json" in c else c)
    except: return None
def download_file(url, p):
    try: 
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, stream=True, timeout=120)
        if r.status_code==200: 
            with open(p, 'wb') as f: 
                for c in r.iter_content(1024): f.write(c)
            return True
    except: pass
    return False

def analyze_layout(p):
    try: return client.chat.completions.create(model=CONFIG.MODEL_VISION, messages=[{"role":"user","content":[{"type":"text","text":CONFIG.PROMPT_DESCRIBE},{"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{_file_to_base64_str(p)}"}}]} ]).choices[0].message.content
    except: return ""

def check_background(p):
    try:
        data = _extract_json(client.chat.completions.create(model=CONFIG.MODEL_VISION, temperature=0.1, messages=[{"role":"user","content":[{"type":"text","text":CONFIG.PROMPT_BG_CHECK},{"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{_file_to_base64_str(p)}"}}]} ]).choices[0].message.content)
        if data: 
            hex_val = data.get("hex_color")
            return {"is_solid": data.get("is_solid", False), "hex_color": hex_val if hex_val else ""}
    except: pass
    return {"is_solid": False, "hex_color": ""}

def generate_image_wrapper(prompt, main_p, use_ref):
    try:
        imgs = []
        if use_ref: 
            for p in CONFIG.REF_IMG_PATHS: 
                if os.path.exists(p): imgs.append(f"data:image/jpeg;base64,{_file_to_base64_str(p)}")
        imgs.append(f"data:image/jpeg;base64,{_file_to_base64_str(main_p)}")
        return client.images.generate(model=CONFIG.MODEL_GEN, prompt=prompt, image=imgs, size=CONFIG.FIXED_GEN_SIZE, response_format="url", watermark=False).data[0].url
    except Exception as e: print(f"Gen Error: {e}"); return None

# === 核心处理 (并发+上传) ===
def _core_process(local_source_path: str, original_filename: str, temp_root_dir: str) -> SingleInputResult:
    print(f"▶️ [处理] {original_filename}")
    work_dir = os.path.join(temp_root_dir, uuid.uuid4().hex); os.makedirs(work_dir, exist_ok=True)
    
    # 1. 基础矫正
    src_p = os.path.join(work_dir, "00_src.jpg"); shutil.copy(local_source_path, src_p)
    corr_p = os.path.join(work_dir, "01_corr.jpg")
    if not perform_correction_safe(src_p, corr_p): shutil.copy(src_p, corr_p)
    
    # 2. 准备并行任务
    executor = ThreadPoolExecutor(max_workers=CONFIG.MAX_WORKERS)

    # 2.0 异步上传原图和矫正图
    # 这样不会阻塞后续的 prompt 分析和生图，上传在后台进行
    fut_upload_src = executor.submit(upload_to_cdn, src_p)
    fut_upload_corr = executor.submit(upload_to_cdn, corr_p)

    # 2.1 背景检测 (异步)
    fut_bg = executor.submit(check_background, corr_p)
    
    # 2.2 视觉分析 (异步)
    fut_layout = executor.submit(analyze_layout, corr_p)

    # 2.3 定义生图+裁切+上传 任务 (✅ 已添加详细计时)
    def run_strategy(strategy, layout_desc_input=None):
        try:
            import time # 引入计时
            t0 = time.time() # 计时开始
            
            prompt = strategy.get_prompt(layout_desc_input)
            tid = f"{strategy.type.value}_{uuid.uuid4().hex[:4]}"
            raw_p = os.path.join(work_dir, f"{tid}_raw.jpg")
            
            # 生图 + 下载
            url = generate_image_wrapper(prompt, corr_p, strategy.use_ref_images)
            if not url or not download_file(url, raw_p): return None
            t1 = time.time() # 生图结束

            # 裁切
            crop_p = os.path.join(work_dir, f"{tid}_crop.jpg")
            ok = strategy.crop_func(raw_p, crop_p)
            fin_p = crop_p if ok else raw_p
            if not ok: shutil.copy(raw_p, crop_p)
            t2 = time.time() # 裁切结束

            # === 上传生成的图片 ===
            print(f"☁️ [{strategy.name}] 开始上传(2张)...")
            url_crop = upload_to_cdn(fin_p)
            url_raw = upload_to_cdn(raw_p)
            t3 = time.time() # 上传结束

            # 打印详细耗时日志
            print(f"⏱️ [{strategy.name}] 耗时详情: 生图+下载{t1-t0:.1f}s | 裁切{t2-t1:.1f}s | 上传{t3-t2:.1f}s | 总计{t3-t0:.1f}s")

            # 返回结果
            return GenerationResult(
                strategy_name=strategy.name,
                crop_image_base64=url_crop, 
                gen_image_base64=url_raw
            )
        except Exception as e:
            print(f"Strategy {strategy.name} Error: {e}")
            return None

    # 3. 提交生图任务
    gen_futures = []

    # 立即提交不依赖视觉分析的任务
    for st in PARALLEL_STRATEGIES:
        if not st.use_layout_analysis:
            gen_futures.append(executor.submit(run_strategy, st, None))

    # 等待视觉分析结果后提交剩余任务
    try:
        layout_res = fut_layout.result(timeout=30)
    except:
        layout_res = ""
    
    for st in PARALLEL_STRATEGIES:
        if st.use_layout_analysis:
            gen_futures.append(executor.submit(run_strategy, st, layout_res))

    # 4. 收集结果
    wait(gen_futures, return_when=ALL_COMPLETED)
    
    results = []
    for f in gen_futures:
        try:
            res = f.result()
            if res: results.append(res)
        except: pass

    try: bg_info = fut_bg.result() 
    except: bg_info = {"is_solid": False, "hex_color": ""}

    # 获取原图上传结果
    try: url_src_res = fut_upload_src.result()
    except: url_src_res = ""
    try: url_corr_res = fut_upload_corr.result()
    except: url_corr_res = ""

    executor.shutdown(wait=False)

    return SingleInputResult(
        filename=original_filename, status="success" if results else "failed",
        original_image_base64=url_src_res,     # 实际上是 URL
        corrected_image_base64=url_corr_res,   # 实际上是 URL
        background_info=BackgroundInfo(**bg_info), generations=results
    )

# === API 路由 ===
@app.post("/restore_batch_file", response_model=BatchRestoreResponse)
def restore_batch_file(files: List[UploadFile] = File(...)):
    if not files: raise HTTPException(400, "No files")
    with tempfile.TemporaryDirectory() as td:
        tasks = []
        for f in files:
            p = os.path.join(td, f.filename)
            with open(p, "wb") as o: shutil.copyfileobj(f.file, o)
            tasks.append((p, f.filename))
        
        res = []
        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = [ex.submit(_core_process, p, n, td) for p, n in tasks]
            for f in futures: res.append(f.result())
            
    return BatchRestoreResponse(total_requested=len(files), total_success=len([r for r in res if r.status=="success"]), batch_results=res)

@app.post("/restore_batch_url", response_model=BatchRestoreResponse)
def restore_batch_url(p: UrlBatchRequest):
    if not p.urls: raise HTTPException(400, "No URLs")
    with tempfile.TemporaryDirectory() as td:
        def dl_and_process(iu): 
            i,u=iu; path=os.path.join(td,f"{i}.jpg")
            return _core_process(path,f"u_{i}",td) if download_file(u,path) else SingleInputResult(filename=u,status="failed")
        
        res = []
        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = [ex.submit(dl_and_process, (i,u)) for i,u in enumerate(p.urls)]
            for f in futures: res.append(f.result())
    return BatchRestoreResponse(total_requested=len(p.urls), total_success=len([r for r in res if r.status=="success"]), batch_results=res)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6002)