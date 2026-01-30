# -*- coding: utf-8 -*-
"""
@File       : image_correct.py
@Author     : Duangang Qu
@Email      : quduangang@outlook.com
@Created    : 2025/9/1 11:57
@Modified   : 2025/12/24 (Optimized for High Resolution)
@Software   : PyCharm
@Description: 图像矫正与增强处理（优化清晰度版）
"""

import base64
import io
import json
import numpy as np
import cv2
import pytesseract
import requests
from PIL import Image, ImageOps
from typing import Union, Optional
import tempfile
import os

# 导入ModelScope相关模块
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# 导入腾讯云SDK相关模块
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.ocr.v20181119 import ocr_client, models

# 导入自定义模块
from textDirectionDetection import text_orientation
from config import TENCENT_CONFIG, TEXTIN_CONFIG, RESNET_CONFIG, IMAGE_CONFIG, SUPPORTED_MODELS


class ImageProcessor:
    """图像处理类，支持多种OCR和图像增强服务"""

    def __init__(self):
        """初始化图像处理器"""
        self.tencent_config = TENCENT_CONFIG
        self.textin_config = TEXTIN_CONFIG
        self.resnet_config = RESNET_CONFIG
        self.image_config = IMAGE_CONFIG

        # 初始化ResNet模型
        self.card_detection_correction = pipeline(
            Tasks.card_detection_correction,
            model=self.resnet_config["MODEL_ID"]
        )

    def process_image(self, image_input: Union[str, Image.Image, np.ndarray],
                      model_name: str, output_path: Optional[str] = None) -> Union[Image.Image, str]:
        """
        处理图像的主接口
        """
        if model_name.lower() not in SUPPORTED_MODELS:
            raise ValueError(f"不支持的模型: {model_name}. 支持的模型: {SUPPORTED_MODELS}")

        # 标准化输入图像
        pil_image, temp_path = self._standardize_input(image_input)

        try:
            # 修复EXIF方向
            pil_image = self._fix_image_orientation(pil_image)

            # 转换为OpenCV格式进行方向校正
            cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            corrected_image = self._correct_text_orientation(cv2_image)

            # [优化1] 保存校正后的图像到临时文件
            # 修改：使用 .png 后缀，避免中间环节的 JPG 有损压缩
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                corrected_temp_path = temp_file.name
                cv2.imwrite(corrected_temp_path, corrected_image)

            # 根据模型类型处理图像
            if model_name.lower() == 'resnet':
                result_image = self._process_with_resnet(corrected_temp_path)
            elif model_name.lower() == 'textin':
                result_image = self._process_with_textin(corrected_temp_path)
            elif model_name.lower() == 'tencent':
                result_image = self._process_with_tencent(corrected_temp_path)

            # 清理临时文件
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
            if os.path.exists(corrected_temp_path):
                os.unlink(corrected_temp_path)

            # [优化2] 最终尺寸调整逻辑
            # 目标尺寸
            target_size = (3000, 1824)
            
            # 使用 LANCZOS 高质量插值算法进行缩放，避免直接 resize 导致的模糊
            if result_image.size != target_size:
                print(f"正在调整尺寸 (LANCZOS): {result_image.size} -> {target_size}")
                result_image = result_image.resize(target_size, Image.Resampling.LANCZOS)

            # 处理输出
            if output_path:
                result_image.save(output_path, 'JPEG', quality=self.image_config.get("JPEG_QUALITY", 95))
                return output_path
            else:
                return result_image

        except Exception as e:
            # 异常发生时也要清理临时文件
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
            if 'corrected_temp_path' in locals() and os.path.exists(corrected_temp_path):
                os.unlink(corrected_temp_path)
            raise Exception(f"图像处理失败: {str(e)}")

    def _standardize_input(self, image_input: Union[str, Image.Image, np.ndarray]) -> tuple:
        """标准化输入图像格式"""
        temp_path = None

        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"图像文件不存在: {image_input}")
            pil_image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            pil_image = image_input.convert('RGB')
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 3:
                image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_input).convert('RGB')
        else:
            raise TypeError(f"不支持的输入类型: {type(image_input)}")

        return pil_image, temp_path

    def _fix_image_orientation(self, img: Image.Image) -> Image.Image:
        """修复图像EXIF元数据中的方向"""
        try:
            return ImageOps.exif_transpose(img)
        except Exception as e:
            print(f"EXIF方向修复失败: {e}")
            return img

    def _correct_text_orientation(self, image_cv2: np.ndarray) -> np.ndarray:
        """使用Tesseract OSD检测文本方向并自动旋转图像"""
        try:
            print("正在检测文本方向...")
            gray_image = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
            _, processed_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            osd = pytesseract.image_to_osd(processed_image, output_type=pytesseract.Output.DICT)

            rotation = osd.get('rotate', 0)
            print(f"检测到的旋转角度: {rotation} 度")

            if rotation != 0:
                print(f"需要旋转，正在校正...")
                if rotation == 90:
                    corrected_image = cv2.rotate(image_cv2, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif rotation == 180:
                    corrected_image = cv2.rotate(image_cv2, cv2.ROTATE_180)
                elif rotation == 270:
                    corrected_image = cv2.rotate(image_cv2, cv2.ROTATE_90_CLOCKWISE)
                else:
                    corrected_image = image_cv2
                print("方向校正完成。")
                return corrected_image
            else:
                print("文本方向正确，无需旋转。")
                return image_cv2
        except Exception as e:
            print(f"文本方向检测失败: {e}")
            return image_cv2

    def _process_with_resnet(self, image_path: str) -> Image.Image:
        """使用ResNet模型处理图像"""
        try:
            print("正在使用ResNet处理图像...")
            result = self.card_detection_correction(image_path)

            if "output_imgs" in result and len(result["output_imgs"]) > 0:
                img = result["output_imgs"][0]
                
                # [优化3] 关键修改：取消中间步骤的强制降采样
                # ❌ 原代码：resized_img = cv2.resize(img, self.image_config["OUTPUT_SIZE"])
                # ✅ 新代码：直接保留模型输出的高清原图
                resized_img = img 

                # 文本方向检测和校正
                label, score = text_orientation(resized_img)
                angle_to_correct = 360 - int(label[0])

                if angle_to_correct == 90:
                    rotated_img = cv2.rotate(resized_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif angle_to_correct == 180:
                    rotated_img = cv2.rotate(resized_img, cv2.ROTATE_180)
                elif angle_to_correct == 270:
                    rotated_img = cv2.rotate(resized_img, cv2.ROTATE_90_CLOCKWISE)
                else:
                    rotated_img = resized_img

                # 转换为PIL Image
                rotated_img_rgb = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB)
                return Image.fromarray(rotated_img_rgb)
            else:
                raise Exception("ResNet处理失败：未返回有效结果")
        except Exception as e:
            raise Exception(f"ResNet处理失败: {str(e)}")

    def _process_with_textin(self, image_path: str) -> Image.Image:
        """使用合合信息处理图像"""
        try:
            print("正在使用合合信息处理图像...")

            headers = {
                'x-ti-app-id': self.textin_config["APP_ID"],
                'x-ti-secret-code': self.textin_config["SECRET_CODE"],
                'Content-Type': 'application/octet-stream'
            }

            with open(image_path, 'rb') as f:
                body = f.read()

            response = requests.post(
                self.textin_config["URL"],
                params=self.textin_config["API_PARAMS"],
                data=body,
                headers=headers
            )
            response.raise_for_status()

            result = json.loads(response.text)

            if result and 'result' in result and result['result']['image_list']:
                image_data = base64.b64decode(result['result']['image_list'][0]["image"])
                img = Image.open(io.BytesIO(image_data))

                # 文本方向检测和校正
                label, score = text_orientation(img)
                resized_img_angle = 360 - int(label[0])

                img = img.rotate(resized_img_angle)
                
                # 这里也可以考虑是否需要移除中间缩放，取决于 textin 返回的图是否已经够大了
                # 如果 textin 返回的本身就是高清的，这里保留 LANCZOS 是对的，但如果中间想保持最大化，可以先去掉
                img_resized = img.resize(self.image_config["OUTPUT_SIZE"], Image.Resampling.LANCZOS)

                return img_resized.convert('RGB')
            else:
                raise Exception(f"合合信息处理失败：API未返回有效图像数据")

        except Exception as e:
            raise Exception(f"合合信息处理失败: {str(e)}")

    def _process_with_tencent(self, image_path: str) -> Image.Image:
        """使用腾讯云处理图像"""
        try:
            print("正在使用腾讯云处理图像...")

            with open(image_path, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode('utf-8')

            enhanced_base64 = self._enhance_image_tencent(img_base64)

            if enhanced_base64:
                enhanced_data = base64.b64decode(enhanced_base64)
                enhanced_img = Image.open(io.BytesIO(enhanced_data))

                # 文本方向检测和校正
                label, score = text_orientation(enhanced_img)
                resized_img_angle = 360 - int(label[0])

                enhanced_img = enhanced_img.rotate(resized_img_angle)
                enhanced_img_resized = enhanced_img.resize(
                    self.image_config["OUTPUT_SIZE"],
                    Image.Resampling.LANCZOS
                )

                return enhanced_img_resized.convert('RGB')
            else:
                raise Exception("腾讯云处理失败：未获取到增强图像")

        except Exception as e:
            raise Exception(f"腾讯云处理失败: {str(e)}")

    def _enhance_image_tencent(self, image_base64: str) -> Optional[str]:
        """调用腾讯云文本图像增强接口"""
        try:
            cred = credential.Credential(
                self.tencent_config["SECRET_ID"],
                self.tencent_config["SECRET_KEY"]
            )
            http_profile = HttpProfile()
            http_profile.endpoint = self.tencent_config["ENDPOINT"]
            client_profile = ClientProfile()
            client_profile.httpProfile = http_profile
            client = ocr_client.OcrClient(cred, self.tencent_config["REGION"], client_profile)

            req = models.ImageEnhancementRequest()
            params = {
                "ImageBase64": image_base64,
                "ReturnImage": "preprocess",
                "TaskType": 1
            }
            req.from_json_string(json.dumps(params))

            resp = client.ImageEnhancement(req)
            return resp.Image
        except TencentCloudSDKException as err:
            print(f"腾讯云API调用失败: {err}")
            return None


processor = ImageProcessor()

def get_Corrected_image(image_path):
    return processor.process_image(image_path, model_name="resnet")

if __name__ == '__main__':
    """
    ImageProcessor 使用示例
    """
    # print("示例: 处理本地图像文件 (ResNet高清优化版)")
    # try:
    #     # 修改为你的实际路径
    #     input_file = "/root/autodl-tmp/image.png"
    #     output_file = "/root/autodl-tmp/image_crop.png"
        
    #     if os.path.exists(input_file):
    #         result_path = processor.process_image(
    #             image_input=input_file,
    #             model_name="resnet",
    #             output_path=output_file
    #         )
    #         print(f"✅ 处理完成! 结果已保存至: {result_path}")
    #         # 可以打印一下最终尺寸确认
    #         with Image.open(result_path) as img:
    #             print(f"📏 最终尺寸: {img.size}")
    #     else:
    #         print(f"❌ 文件不存在: {input_file}")
            
    # except Exception as e:
    #     print(f"❌ 处理失败: {e}")


    print("\n示例: 批量处理图像")
    import os
    
    def batch_process_images(input_dir, output_dir, model_name):
        """批量处理图像"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [f for f in os.listdir(input_dir)
                       if os.path.splitext(f.lower())[1] in supported_extensions]
    
        for i, filename in enumerate(image_files, 1):
            print(f"处理第 {i}/{len(image_files)} 个文件: {filename}")
    
            try:
                input_path = os.path.join(input_dir, filename)
                output_filename = f"{model_name}_{os.path.splitext(filename)[0]}.jpg"
                output_path = os.path.join(output_dir, output_filename)
    
                # 处理图像
                processor.process_image(
                    image_input=input_path,
                    model_name=model_name,
                    output_path=output_path
                )
    
                print(f"  处理完成: {output_path}")
    
            except Exception as e:
                print(f"  处理失败: {e}")
    
    # 批量处理示例（需要修改路径）
    batch_process_images(
        input_dir="/root/autodl-tmp/img",
        output_dir="/root/autodl-tmp/img_only",
        model_name="resnet"
    )