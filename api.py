# -*- coding: utf-8 -*-
"""
@File       : api_server_async.py
@Description: 智能名片翻新 (极速 URL 最终版)
@Logic      : 
    1. 参考图: 固定 URL (零带宽消耗，秒发)
    2. 主图: 矫正后立即上传换 URL (混合加速，省50%带宽)
    3. 画质: 95 (无损高清)
    4. 裁切: 智能双重保障 (红框优先 -> ResNet兜底)
    5. 并发: 暴力全开
@Usage      : nohup python -u api_server_async.py > runtime.log 2>&1 &
"""

import os
import io
import sys
import json
import time
import base64
import asyncio
import logging
import httpx
import cv2
import numpy as np
from typing import List, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from volcenginesdkarkruntime import Ark 
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# 引入优化后的 ResNet 模块
from image_correct_optimized import processor

# ================= 1. 日志配置 =================
logger = logging.getLogger("SmartCard")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.propagate = False

# ================= 2. 全局配置 =================
class CONFIG:
    VOLC_API_KEY = ""
    VOLC_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
    MODEL_GEN = "doubao-seedream-4-5-251128"
    MODEL_VISION = "doubao-seed-1-6-vision-250815"
    
    FIXED_GEN_SIZE = "3000x1824"

    # === 暴力并发配置 ===
    # 由于使用了 URL 模式，带宽压力极小，可以放心拉满
    GPU_SEMAPHORE_LIMIT = 8
    API_SEMAPHORE_LIMIT = 50
    UPLOAD_SEMAPHORE_LIMIT = 50
    WORKFLOW_SEMAPHORE_LIMIT = 15


    # === 上传接口 ===
    UPLOAD_API_URL = "https://tt.36588.com.cn/mcard/common/commonUpload"
    IMG_URL_PREFIX = "https://tt.36588.com.cn/mcard/assets/resource/imgs/normal/"

    REF_LOCAL_DIR = "/home/ubuntu/zj/restore/ref_imgs"
    
    # === [已填入] 参考图 URL 配置 (零带宽消耗) ===
    REF_IMGS_URLS = [
        "https://tt.36588.com.cn/mcard/assets/resource/imgs/normal/printdiy1/M00/B9/17/oYYBAGll6wKAJWQTABAy8xidGMs775.png",
        "https://tt.36588.com.cn/mcard/assets/resource/imgs/normal/printdiy1/M00/B9/18/oYYBAGll6wqAKfDIABUkz0aLF5o094.png",
        "https://tt.36588.com.cn/mcard/assets/resource/imgs/normal/printdiy1/M00/B9/18/oYYBAGll6xCAMoBzAA8pW2TABec002.png",
        "https://tt.36588.com.cn/mcard/assets/resource/imgs/normal/printdiy1/M00/B9/19/oYYBAGll6xWAaDzhAA4BUvSCQIM375.png",
    ]

    # === Prompts ===
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
    图1—图 4为参考图，对图五进行处理：名片，平面设计原稿，矢量图，正视图，图片高清无噪点，扫描仪效果，绝对扁平，无厚度，无遮挡，无折痕，无阴影，无笔迹，无透视，无扭曲，绝对矩形，清晰的文字排版，Adobe Illustrator设计稿，去材质化，高保真，相框，红色边框内。
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

# ================= 3. 系统初始化 =================

app = FastAPI(title="Smart Card Restore Ultimate", description="URL Mode + High Quality")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# 资源池
gpu_lock = asyncio.Semaphore(CONFIG.GPU_SEMAPHORE_LIMIT)
api_lock = asyncio.Semaphore(CONFIG.API_SEMAPHORE_LIMIT)
upload_lock = asyncio.Semaphore(CONFIG.UPLOAD_SEMAPHORE_LIMIT)
workflow_lock = asyncio.Semaphore(CONFIG.WORKFLOW_SEMAPHORE_LIMIT)

cpu_executor = ThreadPoolExecutor(max_workers=64)
ark_client = Ark(api_key=CONFIG.VOLC_API_KEY, base_url=CONFIG.VOLC_BASE_URL)
http_client = httpx.AsyncClient(timeout=60.0, limits=httpx.Limits(max_keepalive_connections=500, max_connections=1000))
img_processor = None
# ✅ [新增] 初始化定时任务调度器
scheduler = AsyncIOScheduler()
# ================= 4. 参考图保活逻辑 (全是新增的) =================

async def ensure_local_refs():
    """确保本地有参考图文件，如果没有则下载初始 URL"""
    if not os.path.exists(CONFIG.REF_LOCAL_DIR):
        os.makedirs(CONFIG.REF_LOCAL_DIR)
    
    for i, url in enumerate(CONFIG.REF_IMGS_URLS):
        file_path = os.path.join(CONFIG.REF_LOCAL_DIR, f"ref_{i}.png")
        if not os.path.exists(file_path):
            logger.info(f"📥 [初始化] 本地缺少参考图 {i+1}，正在从初始 URL 下载备份...")
            content = await async_download(url)
            if content:
                with open(file_path, "wb") as f:
                    f.write(content)
                logger.info(f"✅ [初始化] 参考图 {i+1} 下载成功: {file_path}")
            else:
                logger.error(f"❌ [初始化] 参考图 {i+1} 下载失败! URL: {url}")

async def refresh_reference_images_task():
    """定时任务：读取本地参考图 -> 上传 -> 更新内存 URL"""
    logger.info("⏰ [定时任务] 开始执行参考图保活上传...")
    
    # 1. 确保本地有图
    await ensure_local_refs()
    
    new_urls = []
    success_count = 0
    
    # 2. 遍历本地文件并上传 (假设固定4张)
    for i in range(len(CONFIG.REF_IMGS_URLS)): 
        file_path = os.path.join(CONFIG.REF_LOCAL_DIR, f"ref_{i}.png")
        if os.path.exists(file_path):
            try:
                with open(file_path, "rb") as f:
                    content = f.read()
                
                # 复用上传逻辑
                new_url = await async_upload(content)
                
                if new_url:
                    new_urls.append(new_url)
                    success_count += 1
                    logger.info(f"✅ [定时任务] 参考图 {i+1} 上传成功 -> {new_url}")
                else:
                    logger.error(f"❌ [定时任务] 参考图 {i+1} 上传失败，将保留旧链接")
                    if i < len(CONFIG.REF_IMGS_URLS):
                        new_urls.append(CONFIG.REF_IMGS_URLS[i])
            except Exception as e:
                logger.error(f"❌ [定时任务] 处理参考图 {i+1} 异常: {e}")
                if i < len(CONFIG.REF_IMGS_URLS):
                    new_urls.append(CONFIG.REF_IMGS_URLS[i])
        else:
            logger.error(f"⚠️ [定时任务] 本地文件丢失: {file_path}")
            if i < len(CONFIG.REF_IMGS_URLS):
                new_urls.append(CONFIG.REF_IMGS_URLS[i])

    # 3. 更新全局配置
    if success_count == 4: 
        CONFIG.REF_IMGS_URLS = new_urls
        logger.info(f"🎉 [定时任务] 参考图池已刷新，当前最新 URL 列表: \n{json.dumps(new_urls, indent=2)}")
    else:
        CONFIG.REF_IMGS_URLS = new_urls
        logger.warning(f"⚠️ [定时任务] 参考图刷新完成，但有失败 ({success_count}/4 成功)")

# ================= 修改原来的启动/关闭事件 =================

@app.on_event("startup")
async def startup_event():
    global img_processor
    logger.info("⏳ 正在加载 ResNet 模型(test)...")
    img_processor = processor
    
    # ✅ [优化] 启动定时任务 (加 try-except 保护)
    logger.info("⏰ 正在启动定时任务调度器...")
    try:
        # 1. 立即运行一次保活
        # 注意：这里 await 会阻塞启动，直到下载完成。这是有意为之，确保服务就绪时参考图可用。
        await refresh_reference_images_task() 
    except Exception as e:
        logger.error(f"❌ [启动警告] 初始参考图更新失败，将使用默认或旧缓存: {e}")

    # 2. 添加定时作业：每天 00:00 执行
    scheduler.add_job(refresh_reference_images_task, 'cron', hour=0, minute=0)
    scheduler.start()
    
    logger.info(f"🔥 系统启动 | ...")

@app.on_event("shutdown")
async def shutdown_event():
    await http_client.aclose()
    cpu_executor.shutdown()
    # ✅ [新增] 关闭调度器
    scheduler.shutdown()

# ================= 4. 辅助函数 =================


def _bytes_to_b64_str(data: bytes) -> str:
    b64 = base64.b64encode(data).decode('utf-8')
    return f"data:image/jpeg;base64,{b64}"

def _extract_json(content: str) -> dict:
    try:
        if "```json" in content: content = content.split("```json")[1].split("```")[0]
        elif "```" in content: content = content.split("```")[1].split("```")[0]
        return json.loads(content.strip())
    except: return {}

def _pil_to_base64(img: Image.Image) -> str:
    buff = io.BytesIO()
    img.save(buff, format="JPEG", quality=95) # 保持高清
    return base64.b64encode(buff.getvalue()).decode('utf-8')

def _pil_to_bytes(img: Image.Image) -> bytes:
    buff = io.BytesIO()
    img.save(buff, format="JPEG", quality=95) # 保持高清
    return buff.getvalue()

def _bytes_to_cv2(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def _order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
    return rect

def _try_red_frame_crop_memory(img_bytes: bytes) -> Optional[bytes]:
    try:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_cv is None: return None
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255])),
            cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < 2000: return None
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4:
            pts = _order_points(approx.reshape(4, 2))
            dst = np.array([[0, 0], [2999, 0], [2999, 1823], [0, 1823]], dtype="float32")
            M = cv2.getPerspectiveTransform(pts, dst)
            warped = cv2.warpPerspective(img_cv, M, (3000, 1824))
            return _pil_to_bytes(Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)))
    except: pass
    return None

# ================= 5. 网络功能 =================

async def async_download(url: str) -> Optional[bytes]:
    try:
        resp = await http_client.get(url)
        return resp.content if resp.status_code == 200 else None
    except: return None

async def async_upload(img_bytes: bytes) -> str:
    if not img_bytes: return ""
    async with upload_lock:
        try:
            b64_str = await asyncio.get_event_loop().run_in_executor(
                cpu_executor, lambda: base64.b64encode(img_bytes).decode('utf-8')
            )
            payload = {"base64Str": f"data:image/jpeg;base64,{b64_str}"}
            resp = await http_client.post(CONFIG.UPLOAD_API_URL, json=payload, timeout=180)
            if resp.status_code == 200:
                d = resp.json()
                if d.get("success"): return f"{CONFIG.IMG_URL_PREFIX}{d.get('userData', '')}"
            return ""
        except Exception as e:
            logger.error(f"⚠️ Upload Fail: {e}")
            return ""

# ================= 6. 核心业务 =================

@dataclass
class RestoreStrategy:
    name: str; need_vision: bool; need_ref: bool

STRATEGIES = [
    RestoreStrategy("静态生成", False, False),
    RestoreStrategy("视觉分析", True, False),
    RestoreStrategy("内容锁定", False, True),
    RestoreStrategy("参考图", False, True)
]

# [修改点 3] 核心业务流程重写
# [修改点] 带有详细计时埋点的核心流程
async def process_single_workflow(original_url: str, img_bytes: bytes, filename: str):
    t_start_all = time.time()
    logger.info(f"▶️ [处理] {filename} ({len(img_bytes)/1024:.0f}KB) | 开始计时")

    # ---------------- 1. GPU 矫正 (本地) ----------------
    t0 = time.time()
    async with gpu_lock:
        def _gpu_task(ib):
            cv_img = _bytes_to_cv2(ib)
            if cv_img is None: return None
            # pil_res = img_processor.process_image_memory(cv_img)
            try:
                # process_image 支持直接传入 cv2/numpy 数组，返回 PIL Image
                pil_res = img_processor.process_image(
                    image_input=cv_img, 
                    model_name="resnet"
                )
                return _pil_to_bytes(pil_res)
            except Exception as e:
                logger.error(f"ResNet Process Error: {e}")
                return None
        
        corr_bytes = await asyncio.get_event_loop().run_in_executor(cpu_executor, _gpu_task, img_bytes)
    t_gpu_end = time.time()
    
    if not corr_bytes:
        logger.error(f"❌ ResNet Correct Failed: {filename}")
        return {"filename": filename, "status": "failed_correction"}

    # ---------------- 2. 并行分流 ----------------
    
    # [A 路] 后台上传
    logger.info("☁️ [后台] 启动静默上传矫正图...")
    upload_future = asyncio.create_task(async_upload(corr_bytes))

    # [B 路] 极速转 Base64 (增加耗时打印)
    t_b64_start = time.time()
    corr_base64 = await asyncio.get_event_loop().run_in_executor(
        cpu_executor, _bytes_to_b64_str, corr_bytes
    )
    t_b64_end = time.time()
    
    # [优化] 生成缩略图 Base64 给 Vision 用 (大幅加速视觉分析)
    def _make_low_res_b64(ib):
        try:
            with Image.open(io.BytesIO(ib)) as img:
                img.thumbnail((1024, 1024)) 
                buff = io.BytesIO()
                img.save(buff, format="JPEG", quality=85)
                return _bytes_to_b64_str(buff.getvalue())
        except: return corr_base64
    
    corr_base64_small = await asyncio.get_event_loop().run_in_executor(
        cpu_executor, _make_low_res_b64, corr_bytes
    )

    logger.info(f"📊 [准备阶段] GPU矫正:{t_gpu_end-t0:.2f}s | 转Base64:{t_b64_end-t_b64_start:.2f}s | 原图大小:{len(corr_base64)/1024/1024:.1f}MB")

    # ---------------- 3. AI 调用封装 (增加详细计时) ----------------

    async def call_vision(p, img_input, is_bg_check=False):
        t_req_start = time.time()
        async with api_lock: 
            t_lock_got = time.time() # 拿到锁的时间
            try:
                content_list = [{"type": "text", "text": p}, {"type": "image_url", "image_url": {"url": img_input}}]
                resp = await asyncio.get_event_loop().run_in_executor(
                    cpu_executor, 
                    lambda: ark_client.chat.completions.create(
                        model=CONFIG.MODEL_VISION,
                        messages=[{"role":"user","content": content_list}]
                    ).choices[0].message.content
                )
                t_req_end = time.time()
                # 打印视觉分析耗时
                check_type = "背景检测" if is_bg_check else "布局分析"
                logger.info(f"👁️ [{check_type}] 排队:{t_lock_got-t_req_start:.2f}s | API传输+推理:{t_req_end-t_lock_got:.2f}s")
                
                if is_bg_check: return _extract_json(resp)
                return resp
            except Exception as e:
                logger.error(f"Vision Error: {e}")
                return {} if is_bg_check else ""

    async def call_gen(p, main_img_input, use_ref, strat_name):
        t_req_start = time.time()
        async with api_lock: 
            t_lock_got = time.time()
            try:
                def _run():
                    imgs = CONFIG.REF_IMGS_URLS[:] if use_ref else []
                    imgs.append(main_img_input) 
                    return ark_client.images.generate(
                        model=CONFIG.MODEL_GEN, prompt=p, image=imgs, 
                        size=CONFIG.FIXED_GEN_SIZE, response_format="url", watermark=False
                    ).data[0].url
                
                url = await asyncio.get_event_loop().run_in_executor(cpu_executor, _run)
                t_req_end = time.time()
                
                # [关键] 打印 API 耗时
                logger.info(f"📡 [{strat_name}] 排队:{t_lock_got-t_req_start:.2f}s | API传输+推理:{t_req_end-t_lock_got:.2f}s")
                return url
            except Exception as e:
                logger.error(f"Gen Error: {e}")
                return None

    # ---------------- 4. 启动任务 ----------------
    
    # 视觉任务用缩略图 (small)
    task_layout = asyncio.create_task(call_vision(CONFIG.PROMPT_DESCRIBE, corr_base64_small))
    task_bg = asyncio.create_task(call_vision(CONFIG.PROMPT_BG_CHECK, corr_base64_small, is_bg_check=True))

    async def run_strat(strat, layout_desc=""):
        try:
            t_step0 = time.time()
            prompt = ""
            if strat.name == "视觉分析": prompt = f"{CONFIG.PROMPT_Gen_BASE}\n视觉参考：{layout_desc}"
            elif strat.name == "静态生成": prompt = CONFIG.PROMPT_WithoutVison
            elif strat.name == "内容锁定": prompt = CONFIG.PROMPT_V2STRICT
            elif strat.name == "参考图": prompt = CONFIG.PROMPT_V2SIMPLE
            else: prompt = CONFIG.PROMPT_Gen_BASE

            logger.info(f"🎨 [{strat.name}] 准备请求...")
            
            # 1. 生图 (获取临时 URL)
            gen_temp_url = await call_gen(prompt, corr_base64, strat.need_ref, strat.name)
            if not gen_temp_url: return None
            t_step1 = time.time() # 生图结束

            # 2. 下载 (获取二进制数据)
            gen_bytes = await async_download(gen_temp_url)
            if not gen_bytes: return None
            t_step2 = time.time() # 下载结束

            # =========== 【新增修改 1】启动生成图转存 ===========
            # 拿到 bytes 后，立即在后台启动上传，不阻塞后续的裁切流程
            # 这样裁切和上传是并行的，几乎不增加总耗时
            task_upload_gen = asyncio.create_task(async_upload(gen_bytes))
            # =================================================

            # 3. 裁切 (耗时操作)
            final_crop_bytes = None
            if strat.name in ["内容锁定", "参考图"]:
                final_crop_bytes = await asyncio.get_event_loop().run_in_executor(
                    cpu_executor, _try_red_frame_crop_memory, gen_bytes
                )
            
            if final_crop_bytes is None:
                async with gpu_lock:
                    def _gpu_crop_task(ib):
                        cv_img = _bytes_to_cv2(ib)
                        if cv_img is None:
                            return ib
                        try:
                            pil_res = img_processor.process_image(cv_img, model_name="resnet")
                            return _pil_to_bytes(pil_res)
                        except:
                            return ib

                    final_crop_bytes = await asyncio.get_event_loop().run_in_executor(
                    cpu_executor, _gpu_crop_task, gen_bytes
                        )

            t_step3 = time.time() # 裁切结束
            
            # 4. 上传裁切图
            u_crop = await async_upload(final_crop_bytes)
            
            # =========== 【新增修改 2】等待生成图上传完成 ===========
            # 此时裁切图已经上传完毕，生成图的上传通常也早就完成了
            u_gen_permanent = await task_upload_gen
            
            # 如果上传失败（返回空），为了保险起见，可以回退使用临时 URL，或者直接留空
            # 这里我逻辑设为：如果上传成功用新链接，失败了用豆包临时链接顶一下
            final_gen_url = u_gen_permanent if u_gen_permanent else gen_temp_url
            # ===================================================

            t_step4 = time.time() # 上传结束
            
            # 5. 打印详情
            total_t = t_step4 - t_step0
            api_t = t_step1 - t_step0
            dl_t = t_step2 - t_step1
            crop_t = t_step3 - t_step2
            up_t = t_step4 - t_step3
            
            logger.info(f"✅ [{strat.name}] 总:{total_t:.1f}s | API:{api_t:.1f}s | 下载:{dl_t:.1f}s | 裁切:{crop_t:.1f}s | 上传:{up_t:.1f}s")
            
            return {
                "strategy_name": strat.name,
                "crop_image_url": u_crop,      # 裁切后的永久链接
                "gen_image_url": final_gen_url # 【已修改】生成图的永久链接
            }
        except Exception as e:
            logger.error(f"Strat Error {strat.name}: {e}")
            return None

    # ================= 🚀 修正后的调度逻辑 =================
    
    running_tasks = []

    # 1. 第一梯队：【不需要】视觉分析的任务，立刻 create_task 发车！
    for s in STRATEGIES:
        if not s.need_vision: 
            running_tasks.append(asyncio.create_task(run_strat(s)))
    
    # 2. 中间卡点：等待 Layout 结果
    layout_res = ""
    try: 
        layout_res = await task_layout
    except: pass
        
    # 3. 第二梯队：【需要】视觉分析的任务，拿到结果后发车
    for s in STRATEGIES:
        if s.need_vision: 
            running_tasks.append(asyncio.create_task(run_strat(s, layout_res)))

    gen_results = await asyncio.gather(*running_tasks)
    
    # ---------------- 5. 收尾同步 ----------------
    logger.info("⏳ 正在回收后台上传任务...")
    corr_url_final = await upload_future 
    
    try: bg_info = await task_bg
    except: bg_info = {"is_solid": False, "hex_color": ""}

    logger.info(f"🎉 [结束] {filename} 处理完毕 | 全程耗时: {time.time()-t_start_all:.1f}s")
    
    return {
        "filename": filename,
        "status": "success",
        "original_image_url": original_url,
        "corrected_image_url": corr_url_final,
        "background_info": bg_info, 
        "generations": [r for r in gen_results if r]
    }

# ================= 7. API 路由 =================

class UrlBatchRequest(BaseModel):
    urls: List[str]

@app.post("/restore_batch_url")
async def restore_batch_url(req: UrlBatchRequest):
    logger.info(f"📨 收到 URL 批量请求: {len(req.urls)} 个")
    if not req.urls: raise HTTPException(400, "No URLs")
    
    async def _worker(url, idx):
        # 同样加锁，限制同时下载的图片数量
        async with workflow_lock:
            ib = await async_download(url)
            if not ib: return {"filename": url, "status": "failed_download"}
            return await process_single_workflow(url, ib, f"url_{idx}")
            
    tasks = [_worker(u, i) for i, u in enumerate(req.urls)]
    results = await asyncio.gather(*tasks)
    return {"total": len(req.urls), "success": len([r for r in results if r['status']=='success']), "results": results}

@app.post("/restore_batch_file")
async def restore_batch_file(files: List[UploadFile] = File(...)):
    logger.info(f"📂 收到文件批量请求: {len(files)} 个")
    
    async def _worker(file):
        # [关键] 在读取文件内容之前，先申请“准入证”
        # 只有拿到锁的任务，才允许把文件读入内存，防止 OOM
        async with workflow_lock:
            # 1. 读取文件内容 (只有 15 个任务能同时运行到这里)
            content = await file.read()
            
            # 2. 启动后台上传
            logger.info(f"⬆️ [后台] 原图开始静默上传: {file.filename}")
            task_upload_src = asyncio.create_task(async_upload(content))
            
            # 3. 启动 AI 处理
            process_task = asyncio.create_task(process_single_workflow("", content, file.filename))
            
            # 4. 等待完成
            result = await process_task
            real_src_url = await task_upload_src
            
            # 5. 填补 URL
            if result["status"] == "success":
                result["original_image_url"] = real_src_url
            
            return result

    # 这里虽然创建了所有 task，但它们会在 `async with workflow_lock` 处排队
    # 不会消耗内存去 read() 文件
    tasks = [_worker(f) for f in files]
    results = await asyncio.gather(*tasks)
    return {"total": len(files), "success": len([r for r in results if r['status']=='success']), "results": results}

if __name__ == "__main__":
    import uvicorn
    # 统一使用 6003 端口
    uvicorn.run(app, host="0.0.0.0", port=6003, workers=1)