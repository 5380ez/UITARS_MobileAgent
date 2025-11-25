import base64
import json
import requests


def encode_image(image_path: str) -> str:
    """
    读取本地图片并返回 base64 编码后的字符串。
    在调用模型时，可以拼成：
        f"data:image/png;base64,{encode_image(path)}"
    放到 image_url.url 里。
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def _post_chat_completion(messages, model: str, api_url: str, token: str,
                          max_tokens: int = 2048, temperature: float = 0.0) -> str:
    """
    统一的底层请求封装，适配 Qwen3-VL-Flash 的 OpenAI 兼容接口。
    """
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        # 对于 Qwen / DashScope，Authorization 也是 Bearer <API_KEY>
        "Authorization": f"Bearer {token}",
    }

    data = {
        "model": model,          # 例如: "qwen3-vl-flash"
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        # DashScope 兼容 OpenAI，不强依赖 seed，去掉以减少不兼容风险
        # "seed": 1234,
    }

    try:
        res = requests.post(api_url, headers=headers, data=json.dumps(data), timeout=60)
        res.raise_for_status()
        res_json = res.json()
    except Exception as e:
        # 网络/HTTP错误时直接抛出，让上层决定是否重试
        raise RuntimeError(f"Request to {api_url} failed: {e}") from e

    try:
        # OpenAI 兼容格式: choices[0].message.content
        return res_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Unexpected response format: {res_json}") from e


def inference_chat_uitars(chat, model: str, api_url: str, token: str) -> str:
    """
    适配你现在 UITARS_MobileAgent 里用的那种 chat 结构：
    chat 是一个由 dict 组成的列表：
        [
            {"role": "system", "content": "..."},
            {"role": "user", "content": [...]},  # 图文混合
            ...
        ]
    这里不再做特殊处理，直接把 messages 透传给模型。
    """
    messages = []
    for message in chat:
        messages.append(
            {
                "role": message["role"],
                "content": message["content"],
            }
        )

    return _post_chat_completion(messages, model, api_url, token)


def inference_chat(chat, model: str, api_url: str, token: str) -> str:
    """
    适配老的纯文本对话格式：
        chat = [
            ("system", "xxx"),
            ("user", "xxx"),
            ("assistant", "yyy"),
        ]
    会自动转换成 OpenAI / Qwen 兼容的 messages。
    """
    messages = []
    for role, content in chat:
        messages.append(
            {
                "role": role,
                "content": content,
            }
        )

    return _post_chat_completion(messages, model, api_url, token)
