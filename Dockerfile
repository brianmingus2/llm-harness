FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    gnupg \
    unzip \
    xz-utils \
  && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
  && apt-get install -y --no-install-recommends nodejs \
  && rm -rf /var/lib/apt/lists/*

ARG TARGETARCH
RUN set -eux; \
  arch="x86_64"; \
  if [ "${TARGETARCH}" = "arm64" ]; then arch="aarch64"; fi; \
  curl -fL -o /tmp/typst.tar.xz \
    "https://github.com/typst/typst/releases/download/v0.14.2/typst-${arch}-unknown-linux-musl.tar.xz"; \
  tar -xJf /tmp/typst.tar.xz -C /tmp; \
  mv /tmp/typst-*/typst /usr/local/bin/typst; \
  chmod +x /usr/local/bin/typst; \
  rm -rf /tmp/typst*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m playwright install --with-deps chromium

COPY . .

EXPOSE 3000 8000

RUN chmod +x /app/scripts/run_maxcov.sh

CMD ["/app/scripts/run_maxcov.sh"]
