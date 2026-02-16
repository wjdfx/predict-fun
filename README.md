# Predict.fun Post-Only 双向挂单策略

## 策略规则
- 只使用 `post-only LIMIT` 下单，不使用 MARKET，不做降级重试。
- 每个新市场开始时，同时挂 `UP` 与 `DOWN` 买单。
- 价差映射：
  - 开盘映射偏移：`offset = PriceToBeat_open - Binance_open`
  - 实时映射价：`mapped_ptb = PriceToBeat_now - offset`
  - 价差：`gap = |Binance_Current_now - mapped_ptb|`
- 撤单规则（全程同一阈值）：
  - 若 `end_secs <= force_cancel_seconds`：无条件撤销未完全成交买单。
  - 否则，若 `gap <= gap_keep_threshold_usd`：保留双边买单。
  - 否则（`gap > gap_keep_threshold_usd`）：撤销未完全成交买单。
- 任一方向买单有成交时，立即撤销对手方向未完成买单。

## 安装依赖
```bash
pip install -r requirements.txt
```

## 配置
```bash
cp config.example.yaml config.yaml
```

编辑 `config.yaml`：
- `api.api_key`
- `wallet.private_key`
- 可选 `wallet.predict_account`
- 策略参数：
  - `runtime.gap_keep_threshold_usd`（默认 `0.0008`）
  - `runtime.force_cancel_seconds`（默认 `10`）
  - `runtime.market_guard_seconds`（默认 `30`）
- API/行情参数：
  - `api.base_url`
  - `runtime.current_price_provider`
  - `runtime.binance_symbol`

## 运行
```bash
python bot.py --config config.yaml
```

## 说明
- 该脚本默认在 Binance 读取 `Current Price` 与开盘基准价。
- 若 Predict 市场数据中缺失 `PriceToBeat` 或无法建立映射，除 `<=force_cancel_seconds` 外不会触发价差撤单。
