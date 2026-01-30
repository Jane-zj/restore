import cv2
import numpy as np
from PIL import Image
from paddleocr import DocImgOrientationClassification
from sqlalchemy.dialects.mssql import IMAGE

model = DocImgOrientationClassification(model_name="PP-LCNet_x1_0_doc_ori")


def text_orientation(img):
    """
    对图像进行文字方向分类，自动处理 PIL.Image 和 OpenCV 图像输入。

    参数:
        img (PIL.Image.Image or numpy.ndarray): 输入的图像。
                                                 可以是 PIL Image 对象，
                                                 也可以是 OpenCV 读取的图像 (numpy array)。

    返回:
        tuple: (标签名列表, 置信度列表)
               例如: (['0'], [0.999])
    """

    # --- 开始修改 ---

    # 1. 判断输入图像的类型
    if isinstance(img, Image.Image):
        # 如果是 PIL Image 对象，则进行转换
        # a. 将 PIL 图像转为 NumPy 数组
        # b. 将颜色空间从 RGB (PIL 默认) 转换为 BGR (OpenCV 默认)
        img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    elif isinstance(img, np.ndarray):
        # 如果本身就是 NumPy 数组 (OpenCV 格式)，直接使用
        img_cv2 = img
    else:
        # 如果是其他不支持的类型，可以抛出错误
        raise TypeError(f"不支持的图像输入类型: {type(img)}")

    # --- 结束修改 ---

    # 使用转换后的 OpenCV 格式图像进行预测
    output = model.predict(
        img_cv2,
        batch_size=1
    )

    # 假设返回结构是固定的
    label_name = output[0]["label_names"]
    scores = output[0]["scores"]

    return label_name, scores

def text_orientation_all(img):
    """
    对图像进行文字方向分类，自动处理 PIL.Image 和 OpenCV 图像输入。

    参数:
        img (PIL.Image.Image or numpy.ndarray): 输入的图像。
                                                 可以是 PIL Image 对象，
                                                 也可以是 OpenCV 读取的图像 (numpy array)。

    返回:
        tuple: (标签名列表, 置信度列表)
               例如: (['0'], [0.999])
    """

    # --- 开始修改 ---

    # 1. 判断输入图像的类型
    if isinstance(img, Image.Image):
        # 如果是 PIL Image 对象，则进行转换
        # a. 将 PIL 图像转为 NumPy 数组
        # b. 将颜色空间从 RGB (PIL 默认) 转换为 BGR (OpenCV 默认)
        img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    elif isinstance(img, np.ndarray):
        # 如果本身就是 NumPy 数组 (OpenCV 格式)，直接使用
        img_cv2 = img
    else:
        # 如果是其他不支持的类型，可以抛出错误
        raise TypeError(f"不支持的图像输入类型: {type(img)}")

    # --- 结束修改 ---

    # 使用转换后的 OpenCV 格式图像进行预测
    output = model.predict(
        img_cv2,
        batch_size=1
    )
    print(output)

    # 假设返回结构是固定的
    label_name = output[0]["label_names"]
    scores = output[0]["scores"]

    return label_name, scores


if __name__ == '__main__':
    image_path = r"/Users/quxiaopang/世纪开元/代码/稿件复刻/02-重构版V2/images/466111c113c452a545712ca4a11bfa56.jpg"
    img = cv2.imread(image_path)
    print(text_orientation_all(img))
