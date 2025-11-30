# Docker Compose 使用说明

## 文件结构

- **docker-compose.base.yml** - 基础配置,包含所有共享设置
- **docker-compose.dryrun.yml** - Dry run 模式专用配置
- **docker-compose.live.yml** - 实盘模式专用配置
- **docker-compose.yml** - 原始配置文件(保留作为备份)

## 使用方法

### 启动 Dry Run 模式

```bash
docker compose -f docker-compose.base.yml -f docker-compose.dryrun.yml up -d
```

### 启动实盘模式

```bash
docker compose -f docker-compose.base.yml -f docker-compose.live.yml up -d
```

### 停止服务

```bash
# 停止 Dry Run
docker compose -f docker-compose.base.yml -f docker-compose.dryrun.yml down

# 停止实盘
docker compose -f docker-compose.base.yml -f docker-compose.live.yml down
```

### 查看日志

```bash
# Dry Run 日志
docker compose -f docker-compose.base.yml -f docker-compose.dryrun.yml logs -f

# 实盘日志
docker compose -f docker-compose.base.yml -f docker-compose.live.yml logs -f
```

## 主要区别

### Dry Run 模式
- **容器名称**: 包含 `-dryrun` 后缀
- **数据库**: `tradesv3-dry-run.sqlite`
- **日志文件**: 包含 `-dryrun.log` 后缀
- **API 端口**: 默认 8080 (可通过 `FREQTRADE__API_SERVER__LISTEN_PORT` 配置)
- **命令**: 包含 `--dry-run` 标志

### 实盘模式
- **容器名称**: 包含 `-live` 后缀
- **数据库**: `tradesv3.sqlite`
- **日志文件**: 包含 `-live.log` 后缀
- **API 端口**: 默认 8081 (可通过 `FREQTRADE__API_SERVER__LISTEN_PORT_LIVE` 配置)
- **命令**: 无 `--dry-run` 标志,直接实盘交易

## 同时运行两个模式

由于使用了不同的容器名称、数据库和端口,你可以同时运行 dry run 和实盘模式:

```bash
# 启动 Dry Run
docker compose -f docker-compose.base.yml -f docker-compose.dryrun.yml up -d

# 同时启动实盘
docker compose -f docker-compose.base.yml -f docker-compose.live.yml up -d
```

## 环境变量

在 `.env` 文件中可以配置:

```env
# Dry Run API 端口
FREQTRADE__API_SERVER__LISTEN_PORT=8080

# 实盘 API 端口
FREQTRADE__API_SERVER__LISTEN_PORT_LIVE=8081

# 其他配置
FREQTRADE__BOT_NAME=Example_Test_Account
FREQTRADE__EXCHANGE__NAME=binance
FREQTRADE__TRADING_MODE=futures
FREQTRADE__STRATEGY=NostalgiaForInfinityX7
```

## 简化命令 (可选)

你可以在 `.bashrc` 或 `.zshrc` 中添加别名:

```bash
# Dry Run 别名
alias ft-dryrun-up='docker compose -f docker-compose.base.yml -f docker-compose.dryrun.yml up -d'
alias ft-dryrun-down='docker compose -f docker-compose.base.yml -f docker-compose.dryrun.yml down'
alias ft-dryrun-logs='docker compose -f docker-compose.base.yml -f docker-compose.dryrun.yml logs -f'

# 实盘别名
alias ft-live-up='docker compose -f docker-compose.base.yml -f docker-compose.live.yml up -d'
alias ft-live-down='docker compose -f docker-compose.base.yml -f docker-compose.live.yml down'
alias ft-live-logs='docker compose -f docker-compose.base.yml -f docker-compose.live.yml logs -f'
```

然后你就可以简单地使用:
```bash
ft-dryrun-up    # 启动 dry run
ft-live-up      # 启动实盘
ft-dryrun-logs  # 查看 dry run 日志
ft-live-logs    # 查看实盘日志
```
