#!/usr/bin/env python3
import argparse
import copy
import json
import requests


def resolve_ref(openapi: dict, ref: str) -> dict:
    # ref like "#/components/schemas/ChatCompletionRequest"
    node = openapi
    for p in ref.lstrip("#/").split("/"):
        node = node[p]
    return node


def deref_schema(openapi: dict, schema: dict) -> dict:
    if "$ref" in schema:
        return deref_schema(openapi, resolve_ref(openapi, schema["$ref"]))
    out = copy.deepcopy(schema)
    if "properties" in out:
        for k, v in out["properties"].items():
            out["properties"][k] = deref_schema(openapi, v)
    if "items" in out and isinstance(out["items"], dict):
        out["items"] = deref_schema(openapi, out["items"])
    return out


def collect_defaults(schema: dict, prefix=""):
    defaults = {}
    props = schema.get("properties", {})
    for k, v in props.items():
        key = f"{prefix}.{k}" if prefix else k
        if "default" in v:
            defaults[key] = v["default"]
        # 只展开一层常见结构；深层可按需扩展
        if v.get("type") == "object":
            defaults.update(collect_defaults(v, key))
    return defaults


def get_chat_request_schema(openapi: dict) -> dict:
    path_item = openapi["paths"]["/v1/chat/completions"]["post"]
    schema = path_item["requestBody"]["content"]["application/json"]["schema"]
    return deref_schema(openapi, schema)


def try_get_json(url):
    try:
        r = requests.get(url, timeout=10)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=30002)
    parser.add_argument("--model", default="None")
    parser.add_argument("--context", default="Hello, please introduce yourself.")
    args = parser.parse_args()

    base = f"http://{args.host}:{args.port}"

    # 1) 拉 OpenAPI，提取 chat/completions 请求默认值
    openapi = requests.get(f"{base}/openapi.json", timeout=20).json()
    chat_schema = get_chat_request_schema(openapi)
    schema_defaults = collect_defaults(chat_schema)

    # 2) 你显式发送的最小请求
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.context}],
    }

    # 3) 发请求
    resp = requests.post(f"{base}/v1/chat/completions", json=payload, timeout=120)
    obj = resp.json()

    # 4) 打印“显式参数 vs schema默认值”
    explicit_keys = set(payload.keys())
    default_candidates = {
        k: v for k, v in schema_defaults.items()
        if "." not in k and k not in explicit_keys
    }

    print("\n=== Explicit payload (you sent) ===")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    print("\n=== OpenAPI-declared defaults for omitted top-level fields ===")
    print(json.dumps(default_candidates, ensure_ascii=False, indent=2))

    # 5) 打印运行结果中可观测到的“服务端行为”
    print("\n=== Observable server behavior from response ===")
    choice0 = (obj.get("choices") or [{}])[0]
    usage = obj.get("usage", {})
    print(json.dumps({
        "finish_reason": choice0.get("finish_reason"),
        "matched_stop": choice0.get("matched_stop"),
        "usage": usage,
        "metadata": obj.get("metadata", {}),
    }, ensure_ascii=False, indent=2))

    # 6) 可选信息：server/model info
    model_info = try_get_json(f"{base}/get_model_info")
    server_info = try_get_json(f"{base}/get_server_info")
    print("\n=== /get_model_info ===")
    print(json.dumps(model_info, ensure_ascii=False, indent=2))
    print("\n=== /get_server_info ===")
    print(json.dumps(server_info, ensure_ascii=False, indent=2))

    print(
        "\n[NOTE] 以上能看到“显式参数 + 文档默认值 + 结果侧证据”。"
        "如果你要100%拿到“最终resolved sampling params”，需要在服务端chat handler里打印解析后的sampling params。"
    )


if __name__ == "__main__":
    main()