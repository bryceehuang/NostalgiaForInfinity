# Freqtrade 多交易所多模式运行指南

本指南详细说明了如何使用 Docker Compose 同时运行 Binance 和 OKX 的 Dry Run（模拟）与 Live（实盘）机器人。

## 1. 架构说明

系统采用**组合式配置**，通过叠加多个 YAML 文件来实现灵活部署：

*   **`docker-compose.base.yml`**: 核心基础配置（镜像、路径映射、网络代理）。
*   **交易所配置**:
    *   `docker-compose.binance.yml`: Binance 特定设置及 `.env.binance` 关联。
    *   `docker-compose.okx.yml`: OKX 特定设置及 `.env.okx` 关联。
*   **运行模式配置**:
    *   `docker-compose.dryrun.yml`: 开启模拟交易，指定专用端口与数据库。
    *   `docker-compose.live.yml`: 开启实盘交易，指定专用端口与数据库。

---

## 2. 环境配置文件准备

你需要为每个交易所准备独立的 API 密钥文件。项目根目录下提供了 `.example` 模板。

### Binance 配置
1. 复制模板：`cp env.binance.example .env.binance`
2. 编辑 `.env.binance`：
   * 填写 `FREQTRADE__EXCHANGE__KEY`
   * 填写 `FREQTRADE__EXCHANGE__SECRET`

### OKX 配置
1. 复制模板：`cp env.okx.example .env.okx`
2. 编辑 `.env.okx`：
   * 填写 `FREQTRADE__EXCHANGE__KEY`
   * 填写 `FREQTRADE__EXCHANGE__SECRET`
   * 填写 `FREQTRADE__EXCHANGE__PASSWORD` (OKX Passphrase)

---

## 3. 运行命令总结

为了实现完全的隔离（防止孤儿容器警告），建议为每个交易所指定一个项目名称 (`-p`)。

### Binance 机器人
| 模式 | 运行命令 | Web UI 地址 |
| :--- | :--- | :--- |
| **Dry Run** | `docker compose -p ft_binance -f docker-compose.base.yml -f docker-compose.binance.yml -f docker-compose.dryrun.yml up -d binance` | `http://localhost:8080` |
| **Live** | `docker compose -p ft_binance -f docker-compose.base.yml -f docker-compose.binance.yml -f docker-compose.live.yml up -d binance` | `http://localhost:8082` |

### OKX 机器人
| 模式 | 运行命令 | Web UI 地址 |
| :--- | :--- | :--- |
| **Dry Run** | `docker compose -p ft_okx -f docker-compose.base.yml -f docker-compose.okx.yml -f docker-compose.dryrun.yml up -d okx` | `http://localhost:8081` |
| **Live** | `docker compose -p ft_okx -f docker-compose.base.yml -f docker-compose.okx.yml -f docker-compose.live.yml up -d okx` | `http://localhost:8083` |

---

## 4. 日志与监控

### 查看实时日志
*   **Binance**: `docker compose -p ft_binance -f docker-compose.binance.yml logs -f binance`
*   **OKX**: `docker compose -p ft_okx -f docker-compose.okx.yml logs -f okx`

### 检查运行状态
*   使用标准 Docker 命令：`docker ps`
*   查看指定项目状态：`docker compose -p ft_binance ps`

---

## 5. 存储说明 (user_data)

每个实例使用独立的 SQLite 数据库文件，确保数据互不干扰：

*   **Binance Dry Run**: `tradesv3_binance_dryrun.sqlite`
*   **Binance Live**: `tradesv3_binance_live.sqlite`
*   **OKX Dry Run**: `tradesv3_okx_dryrun.sqlite`
*   **OKX Live**: `tradesv3_okx_live.sqlite`

日志文件存储在 `user_data/logs/` 目录下。

---

## 6. 进阶管理 (别名设置)

为了方便操作，建议在你的 `.zshrc` 或 `.bashrc` 中添加别名：

```bash
# Binance 管理
alias ft-bin-dry='docker compose -p ft_binance -f docker-compose.base.yml -f docker-compose.binance.yml -f docker-compose.dryrun.yml'
alias ft-bin-live='docker compose -p ft_binance -f docker-compose.base.yml -f docker-compose.binance.yml -f docker-compose.live.yml'

# OKX 管理
alias ft-okx-dry='docker compose -p ft_okx -f docker-compose.base.yml -f docker-compose.okx.yml -f docker-compose.dryrun.yml'
alias ft-okx-live='docker compose -p ft_okx -f docker-compose.base.yml -f docker-compose.okx.yml -f docker-compose.live.yml'
```


---

## 7. JSON 配置文件说明

默认情况下，机器人会根据环境变量自动加载配置。如果你需要为不同实例指定不同的 `config.json`，有以下两种方式：

### 方式 A：通过环境变量指定（推荐）
在 `.env.binance` 或 `.env.okx` 中添加：
```env
FREQTRADE__CONFIG=configs/your_custom_config.json
```
注意：路径必须是容器内的路径（由于挂载了 `./configs`，所以以 `configs/` 开头）。

### 方式 B：修改 Compose 文件
在 `docker-compose.dryrun.yml` 或 `docker-compose.live.yml` 的 `command` 部分手动添加 `--config` 参数：
```yaml
    command: >
      trade
      --config configs/config.json
      --config configs/pairlist-static-binance.json
      ...
```

---

## 8. 常见问题 (FAQ)

*   **端口占用**: 如果 8080-8083 端口已被占用，请在启动命令前通过环境变量修改，例如：`PORT_BINANCE_DRY=9000 docker compose ... up -d`。
*   **代理设置**: 基础配置中默认启用了 `host.docker.internal:7897` 代理。如果你的环境不需要代理，请注释掉 `docker-compose.base.yml` 中的 `environment` 部分。
