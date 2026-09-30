# 学习笔记：把后端打包成镜像，部署到 VKube container service
# - 构建流程：Mac 上 buildx 交叉编译出 linux/amd64 镜像（服务器 CPU 架构和 Apple 芯片不同）→ 推送到 Docker Hub
#   → container service 拉取运行，配置全部以环境变量注入（参考 .env.example）。
# - 分层缓存：先只复制 pyproject.toml / uv.lock 安装依赖，再复制代码。改代码不会触发重新安装依赖，构建快很多。
# - 镜像里不包含任何密钥（.dockerignore 排除了 .env*），也不包含文档：容器启动时由 entrypoint.sh 克隆文档。
# - 安全：用非 root 用户 app 运行。
# - HEALTHCHECK：平台据此判断容器是否健康。
# - 只开 1 个 worker：限流计数存在进程内存里（见 app/ratelimit.py），多开 worker 限流会失效。
# - --proxy-headers：容器前面有平台的反向代理（负责 HTTPS），让应用能拿到真实的协议和客户端 IP。
#
# Build for the server's CPU, not the Mac's:
#   docker buildx build --platform linux/amd64 -t <dockerhub-user>/vkube-rag:0.1.0 --push .
FROM python:3.12-slim

# git: the entrypoint clones the docs repo and /api/sync pulls it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# From PyPI rather than ghcr.io/astral-sh/uv, which is not reachable from every network.
RUN pip install --no-cache-dir uv==0.10.9

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Dependencies first, so code changes don't reinstall them.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app
COPY scripts ./scripts
COPY docker/entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh \
    && useradd --create-home --uid 1000 app \
    && mkdir -p /app/data \
    && chown -R app:app /app/data
USER app

EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/api/health', timeout=4)"

ENTRYPOINT ["/entrypoint.sh"]
# One worker on purpose: rate-limit counters live in process memory.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--proxy-headers", "--forwarded-allow-ips", "*"]
