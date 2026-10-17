# Stage 1: build the wasm bundle with the same tool versions as mise.toml
FROM docker.io/library/debian:trixie-slim@sha256:a29215f6a35e51e22adffa17f89e9d2ef06214e64a2bad10d765c46aea49f11f AS build

# gcc + libc6-dev is all Rust build scripts need on the host side.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl gcc libc6-dev \
 && rm -rf /var/lib/apt/lists/*

ENV MISE_VERSION=v2026.10.3 \
    MISE_YES=1 \
    PATH=/root/.local/bin:/root/.local/share/mise/shims:$PATH
RUN curl -fsSL https://mise.run | sh

WORKDIR /src
COPY mise.toml ./
RUN mise trust && mise install

COPY Cargo.toml Cargo.lock Trunk.toml index.html ./
COPY src ./src
# Cache mounts keep compiled deps and trunk's wasm-bindgen/wasm-opt downloads
# across builds without baking them into a layer.
RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cargo/git \
    --mount=type=cache,target=/root/.cache/trunk \
    --mount=type=cache,target=/src/target \
    mise run build \
 && gzip -k -9 dist/*.js dist/*.wasm

# Stage 2: serve the static files as non-root on 8080
FROM docker.io/nginxinc/nginx-unprivileged:1.29-alpine@sha256:0c79d56aee561a1d81c63f00eee5fb5fe29279560cdc55e91425133104c7fbe6
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /src/dist /usr/share/nginx/html
EXPOSE 8080
