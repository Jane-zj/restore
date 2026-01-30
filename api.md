# 第一阶段

## 重建

### 1. 基础信息

| 项目 | 说明 |
| --- | --- |
| **Base URL** | `https://u284779-a567-60ee3444.nmb1.seetacloud.com:8443`|
| **协议** | HTTP / HTTPS |
| **数据格式** | JSON |
| **字符编码** | UTF-8 |

### 2. 核心逻辑说明

**全并发执行 (Parallel Execution)**：
系统接收请求后，会针对每一张图片 **同时启动 4 种生成策略**。

#### 包含的生成策略

| 策略名称 | 代码标识 | 逻辑描述 | 裁剪算法 |
| --- | --- | --- | --- |
| **静态生成** | `V7_STATIC` | 纯 Prompt 控制，不进行耗时的视觉分析，速度最快。 | ResNet 算法 |
| **视觉分析** | `V7_DYNAMIC` | 先调用 Vision 模型反推布局描述，再生成，还原度更高。 | ResNet 算法 |
| **内容锁定** | `V8_STRICT` | 强参考图模式，严格锁定内容，抑制幻觉。 | 红框检测算法 |
| **参考图** | `V8_SIMPLE` | 弱参考图模式，兼顾风格参考。 | 红框检测算法 |

---

### 3. 接口详情

#### 3.1 URL 批量名片重建

提交图片 URL 列表，服务器自动下载并并发处理。

* **URL**: `/restore_batch_url`
* **Method**: `POST`
* **Content-Type**: `application/json`

##### 请求参数

| 参数名 | 类型 | 必填 | 描述 | 示例 |
| --- | --- | --- | --- | --- |
| `urls` | Array[String] | 是 | 图片下载链接列表 | `["http://site.com/1.jpg", "http://site.com/2.png"]` |

##### 响应示例 (Success 200)

```json
{
  "total_requested": 1,
  "total_success": 1,
  "batch_results": [
    {
      "filename": "u_0",
      "status": "success",
      "error_msg": "",
      "original_image_base64": "/9j/4AAQSk...",  // 原始上传图
      "corrected_image_base64": "/9j/4AAQ...",   // 经 ResNet 预矫正后的参考图
      "background_info": {
        "is_solid": true,
        "hex_color": "#FFFFFF"
      },
      "generations": [
        {
          "strategy_name": "静态生成",
          "crop_image_base64": "/9j/4AAQ...",    // 最终交付图
          "gen_image_base64": "/9j/4AAQ..."      // AI 原始生成大图 
        },
        {
          "strategy_name": "视觉分析",
          "crop_image_base64": "/9j/4AAQ...",
          "gen_image_base64": "/9j/4AAQ..."
        },
        {
          "strategy_name": "内容锁定",
          "crop_image_base64": "/9j/4AAQ...",
          "gen_image_base64": "/9j/4AAQ..."
        },
        {
          "strategy_name": "参考图",
          "crop_image_base64": "/9j/4AAQ...",
          "gen_image_base64": "/9j/4AAQ..."
        }
      ]
    }
  ]
}

```

---

#### 3.2 文件批量名片重建

直接上传本地图片文件（二进制流），服务器接收后并发处理。

* **URL**: `/restore_batch_file`
* **Method**: `POST`
* **Content-Type**: `multipart/form-data`

##### 请求参数

| 参数名 | 类型 | 必填 | 描述 |
| --- | --- | --- | --- |
| `files` | File List | 是 | 支持多文件同时上传，Form Key 为 `files` |

##### 响应结构

响应 JSON 结构与 `/restore_batch_url` 完全一致。

---

### 4. 字段字典

#### 4.1 顶层响应字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `total_requested` | int | 本次请求处理的总图片数量 |
| `total_success` | int | 处理成功的图片数量  |
| `batch_results` | Array | 结果列表，包含 `SingleInputResult` 对象 |

#### 4.2 单图结果对象 (`SingleInputResult`)

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `filename` | string | 文件名或任务标识 |
| `status` | string | 任务状态 (`success` 或 `failed`) |
| `error_msg` | string | 错误信息（仅在 `status` 为 `failed` 时有值） |
| `original_image_base64` | string | 用户上传的原始图片 (Base64) |
| `corrected_image_base64` | string | 经本地 ResNet 预矫正后的真值参考图 (Base64) |
| `background_info` | Object | 背景分析结果 (见下表) |
| `generations` | Array | 生成方案列表，包含 `GenerationResult` 对象 |

#### 4.3 背景信息对象 (`BackgroundInfo`)

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `is_solid` | boolean | `true`: 纯色/单色背景; `false`: 复杂背景 |
| `hex_color` | string | 如果是纯色，返回十六进制颜色码 (如 `#FFFFFF`)，否则为空字符串 |

#### 4.4 生成方案对象 (`GenerationResult`)

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `strategy_name` | string | 策略名称，固定为以下四种之一：<br>

<br>1. `静态生成`<br>

<br>2. `视觉分析`<br>

<br>3. `内容锁定`<br>

<br>4. `参考图` |
| `crop_image_base64` | string | **最终交付图** (已根据策略自动裁切白边/背景，即成品图) |
| `gen_image_base64` | string | AI 生成的原始大图 (未经裁切，包含背景，仅供调试或兜底) |

---

### 5. 状态码与错误处理

| HTTP 状态码 | 描述 |
| --- | --- |
| `200` | 请求成功 (业务逻辑成功与否需检查 body 中的 `status`) |
| `400` | 参数错误 (如 URL 列表为空、未上传文件) |
| `422` | 数据校验错误 (JSON 格式不符) |
| `500` | 服务器内部错误 (如下载超时、模型服务不可用) |

#### 错误响应示例 (400 Bad Request)

```json
{
  "detail": "No files"
}

```

## 手动矫正

### 1. 名片精确透视矫正

**接口说明**
该接口用于对上传的原始图片进行基于四点坐标的精确透视变换。常用于当自动检测失败或用户需要微调裁剪范围时，通过前端传递确定的四个角点坐标，后台生成矫正后的高清名片图。

**基本信息**

| 项目 | 内容 |
| --- | --- |
| **接口地址** | `https://uu284779-a567-60ee3444.nmb1.seetacloud.com:8443/api/correct` |
| **请求方法** | `POST` |
| **Content-Type** | `multipart/form-data` |
| **响应格式** | `JSON` |

#### 请求参数

| 参数名 | 类型 | 必填 | 描述 | 示例 |
| --- | --- | --- | --- | --- |
| `file` | File | **是** | 原始图片文件 (支持 JPG/PNG 等常见格式) | (二进制文件流) |
| `points` | String | **是** | 包含四个角点坐标的 **JSON 字符串**。<br>

<br>

<br>⚠️ **重要规则**：<br>

<br>1. 必须严格按照 **[左上, 右上, 右下, 左下]** 的顺时针顺序传入。<br>

<br>2. 格式必须为二维数组字符串 `[[x,y], [x,y], [x,y], [x,y]]`。 | `[[100,100], [500,100], [500,500], [100,500]]` |

#### 响应示例 (Success 200)

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "ordered_corners": [
      [100.0, 100.0],
      [500.0, 100.0],
      [500.0, 500.0],
      [100.0, 500.0]
    ],
    "corrected_image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD...",
    "size": {
      "width": 400,
      "height": 400
    }
  }
}

```

#### 响应字段说明

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `code` | int | 业务状态码 (200 表示处理成功) |
| `message` | string | 状态描述信息 |
| `data` | Object | 核心返回数据 |
| └ `ordered_corners` | Array | 后端接收并确认使用的坐标点列表，顺序固定为：左上、右上、右下、左下 |
| └ `corrected_image` | string | 矫正后的图片数据 (Base64格式)，带 `data:image/jpeg;base64` 前缀，可直接用于 `src` 属性 |
| └ `size` | Object | 矫正后的图片尺寸信息 |
|    └ `width` | int | 图片宽度 (px) |
|    └ `height` | int | 图片高度 (px) |

#### 错误响应示例

**参数格式错误 (400)**

```json
{
  "detail": "Points 格式错误"
}

```

**坐标点数量不足 (400)**

```json
{
  "detail": "必须传入 4 个坐标点"
}

```

#### 状态码说明

| 状态码 | 说明 |
| --- | --- |
| `200` | 请求成功 (需进一步检查 response body 中的 `code` 确认业务逻辑是否成功) |
| `400` | 请求参数错误 (如 `points` 解析失败、坐标点不足4个、图片文件损坏) |
| `422` | 数据校验错误 (缺少必填字段 `file` 或 `points`) |
| `500` | 服务器内部错误 (如图像处理算法异常、编码失败) |

#### cURL 调用示例

```bash
curl -X POST "http://127.0.0.1:6008/api/correct" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/your/image.jpg" \
     -F "points=[[100, 100], [500, 100], [500, 500], [100, 500]]"

```