#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml
from eth_account import Account
from eth_account.messages import encode_defunct
from zoneinfo import ZoneInfo

from predict_sdk import OrderBuilder, OrderBuilderOptions
from predict_sdk.types import BuildOrderInput, LimitHelperInput, Side
from strategy_rules import evaluate_cancel_decision

WAD = Decimal("1000000000000000000")
ET = ZoneInfo("America/New_York")
POST_ONLY_ERROR_HINTS = [
    "post only",
    "post-only",
    "postonly",
    "maker only",
    "maker-only",
    "would execute",
    "take liquidity",
    "cross",
]


def to_wei(value: Decimal) -> int:
    return int((value * WAD).quantize(Decimal("1")))


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def parse_decimal(v: Any, field_name: str) -> Decimal:
    try:
        return Decimal(str(v))
    except (InvalidOperation, TypeError) as exc:
        raise ValueError(f"invalid decimal for {field_name}: {v}") from exc


def validate_private_key(pk: str) -> None:
    s = (pk or "").strip()
    if s.startswith("0x"):
        s = s[2:]
    if len(s) == 40:
        raise ValueError(
            "wallet.private_key looks like an address (20 bytes). "
            "Please provide a 32-byte EVM private key (64 hex chars, usually with 0x prefix)."
        )
    if len(s) != 64:
        raise ValueError("wallet.private_key must be 32 bytes (64 hex chars, optional 0x prefix).")
    try:
        int(s, 16)
    except ValueError as exc:
        raise ValueError("wallet.private_key is not valid hex") from exc


def parse_iso_ts(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts).astimezone(timezone.utc)
    except ValueError:
        return None


def to_plain_dict(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {k: to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_plain_dict(i) for i in obj]
    return obj


def maybe_get(d: Dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def normalize_signed_order(signed: Any, order_hash: str) -> Dict[str, Any]:
    raw = to_plain_dict(signed)
    order_obj = raw.get("order", raw)
    signature = maybe_get(raw, "signature", "sig")

    return {
        "maker": maybe_get(order_obj, "maker", "makerAddress"),
        "signer": maybe_get(order_obj, "signer", "signerAddress")
        or maybe_get(raw, "signer")
        or maybe_get(order_obj, "maker", "makerAddress"),
        "taker": maybe_get(order_obj, "taker", "takerAddress") or "0x0000000000000000000000000000000000000000",
        "tokenId": str(maybe_get(order_obj, "tokenId", "token_id")),
        "makerAmount": str(maybe_get(order_obj, "makerAmount", "maker_amount")),
        "takerAmount": str(maybe_get(order_obj, "takerAmount", "taker_amount")),
        "side": int(maybe_get(order_obj, "side", "order_side")),
        "feeRateBps": str(maybe_get(order_obj, "feeRateBps", "fee_rate_bps") or "0"),
        "nonce": str(maybe_get(order_obj, "nonce") or "0"),
        "expiration": str(maybe_get(order_obj, "expiration") or "0"),
        "salt": str(maybe_get(order_obj, "salt") or "0"),
        "signatureType": int(maybe_get(order_obj, "signatureType", "signature_type") or 0),
        "signature": signature,
        "hash": order_hash,
    }


def market_sort_key(m: Dict[str, Any]) -> Tuple[int, str]:
    c = parse_iso_ts(str(maybe_get(m, "createdAt", "created_at", "openedAt", "opened_at") or ""))
    if c:
        return int(c.timestamp()), str(maybe_get(m, "id", ""))
    return 0, str(maybe_get(m, "id", ""))


def parse_market_times(market: Dict[str, Any], duration_minutes: int) -> Tuple[Optional[datetime], Optional[datetime]]:
    open_ts = parse_iso_ts(str(maybe_get(market, "openTime", "open_time", "startsAt", "startTime", "openedAt", "createdAt") or ""))
    close_ts = parse_iso_ts(str(maybe_get(market, "closeTime", "close_time", "endsAt", "endTime", "closedAt") or ""))
    if open_ts and close_ts:
        return open_ts, close_ts

    slug = str(maybe_get(market, "slug", "marketSlug", "market_slug") or "")
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-15-minutes", slug)
    if m:
        dt = datetime(
            int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)), tzinfo=ET
        ).astimezone(timezone.utc)
        return dt, dt + timedelta(minutes=duration_minutes)

    if open_ts and not close_ts:
        return open_ts, open_ts + timedelta(minutes=duration_minutes)

    return None, None


def get_market_title(market: Dict[str, Any]) -> str:
    return str(maybe_get(market, "title", "question") or "")


def extract_order_id_and_hash(resp: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    data = resp.get("data", {}) if isinstance(resp, dict) else {}
    oid = maybe_get(data, "id", "orderId", "order_id") or maybe_get(resp, "id", "orderId", "order_id")
    ohash = maybe_get(data, "hash", "orderHash", "order_hash") or maybe_get(resp, "hash", "orderHash", "order_hash")
    return (str(oid) if oid is not None else None, str(ohash) if ohash is not None else None)


def extract_order_status_and_filled_wei(resp: Dict[str, Any]) -> Tuple[str, int]:
    data = resp.get("data", {}) if isinstance(resp, dict) else {}
    src = data if isinstance(data, dict) else (resp if isinstance(resp, dict) else {})
    status = str(maybe_get(src, "status") or "UNKNOWN").upper()
    filled_raw = str(
        maybe_get(
            src,
            "amountFilled",
            "amount_filled",
            "filledAmount",
            "filled_amount",
            "filled",
            "filledShares",
            "filled_shares",
            "executedAmount",
            "executed_amount",
            "matchedAmount",
            "matched_amount",
            "sizeFilled",
            "size_filled",
        )
        or "0"
    )
    try:
        filled_wei = int(Decimal(filled_raw))
    except (InvalidOperation, ValueError):
        filled_wei = 0
    return status, filled_wei


def extract_market_prices(market: Dict[str, Any]) -> Tuple[str, str]:
    ptb = maybe_get(
        market,
        "priceToBeat",
        "price_to_beat",
        "priceTarget",
        "price_target",
        "targetPrice",
        "target_price",
    )
    variant = market.get("variantData") if isinstance(market.get("variantData"), dict) else {}
    if ptb is None and isinstance(variant, dict):
        ptb = maybe_get(variant, "startPrice", "start_price")

    current = maybe_get(
        market,
        "currentPrice",
        "current_price",
        "indexPrice",
        "index_price",
        "referencePrice",
        "reference_price",
        "oraclePrice",
        "oracle_price",
    )
    if current is None and isinstance(variant, dict):
        current = maybe_get(variant, "endPrice", "end_price")

    return (str(ptb) if ptb is not None else "n/a", str(current) if current is not None else "n/a")


def pick_up_down_tokens(market: Dict[str, Any]) -> Tuple[str, str]:
    outcomes = maybe_get(market, "outcomes", "market_outcomes") or []
    up_token = None
    down_token = None

    for o in outcomes:
        name = str(maybe_get(o, "name", "outcome", "label") or "").lower()
        token = maybe_get(o, "onChainId", "on_chain_id", "tokenId", "token_id", "id")
        if token is None:
            continue
        token_str = str(token)

        if any(k in name for k in ["up", "yes"]):
            up_token = token_str
        elif any(k in name for k in ["down", "no"]):
            down_token = token_str

    if (up_token is None or down_token is None) and len(outcomes) == 2:
        t0 = str(maybe_get(outcomes[0], "onChainId", "tokenId", "id"))
        t1 = str(maybe_get(outcomes[1], "onChainId", "tokenId", "id"))
        if up_token is None:
            up_token = t0
        if down_token is None:
            down_token = t1

    if not up_token or not down_token:
        raise ValueError("failed to resolve Up/Down outcome token ids")

    return up_token, down_token


@dataclass
class LocalOrder:
    outcome: str
    token_id: str
    side: str
    price_per_share: Decimal
    quantity_wei: int
    order_hash: str
    order_id: Optional[str] = None
    created_at: datetime = field(default_factory=now_utc)
    cancel_requested: bool = False
    cancel_attempts: int = 0
    last_cancel_attempt_at: Optional[datetime] = None
    last_known_status: str = "UNKNOWN"
    last_known_filled_wei: int = 0
    cancel_finalized: bool = False
    cancel_missing_polls: int = 0


@dataclass
class MarketCycle:
    market_id: str
    market_title: str
    market_slug: str
    start_at: Optional[datetime]
    end_at: Optional[datetime]
    buy_orders: List[LocalOrder] = field(default_factory=list)
    market_detail: Dict[str, Any] = field(default_factory=dict)
    market_detail_fetched_at: Optional[datetime] = None
    binance_open: Optional[str] = None
    binance_open_time: Optional[datetime] = None
    ptb_open: Optional[str] = None
    ptb_binance_offset: Optional[str] = None
    gap_rule_cancelled: bool = False


class PredictApi:
    def __init__(self, base_url: str, api_key: str, timeout: int = 20):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        self.jwt_token: Optional[str] = None

    def _headers(self, with_auth: bool = True) -> Dict[str, str]:
        h = {
            "x-api-key": self.api_key,
            "content-type": "application/json",
        }
        if with_auth and self.jwt_token:
            h["authorization"] = f"Bearer {self.jwt_token}"
        return h

    def _request(self, method: str, path: str, *, params=None, data=None, auth=True) -> Any:
        url = f"{self.base_url}{path}"
        resp = self.session.request(
            method=method,
            url=url,
            headers=self._headers(with_auth=auth),
            params=params,
            data=json.dumps(data) if data is not None else None,
            timeout=self.timeout,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"{method} {path} failed: {resp.status_code} {resp.text[:300]}")
        if not resp.text:
            return {}
        return resp.json()

    def get_markets(self, first: int, status: str = "OPEN") -> List[Dict[str, Any]]:
        resp = self._request("GET", "/v1/markets", params={"first": first, "status": status})
        return resp.get("data", []) or []

    def get_market(self, market_id: str) -> Dict[str, Any]:
        resp = self._request("GET", f"/v1/markets/{market_id}")
        data = resp.get("data")
        if isinstance(data, dict):
            return data
        return resp if isinstance(resp, dict) else {}

    def get_orders(self, first: int = 200, status: Optional[str] = None) -> List[Dict[str, Any]]:
        safe_first = max(1, min(int(first), 150))
        params: Dict[str, Any] = {"first": safe_first}
        if status:
            params["status"] = status
        resp = self._request("GET", "/v1/orders", params=params)
        return resp.get("data", []) or []

    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        resp = self._request("GET", f"/v1/orders/{order_id}")
        data = resp.get("data")
        if isinstance(data, dict):
            return data
        if isinstance(resp, dict) and maybe_get(resp, "id", "orderId", "order_id") is not None:
            return resp
        return None

    def create_order(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/v1/orders", data=payload)

    def remove_orders(self, ids: List[str]) -> Dict[str, Any]:
        return self._request("POST", "/v1/orders/remove", data={"data": {"ids": ids}})


class PredictTrader:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config

        api_cfg = config["api"]
        wallet_cfg = config["wallet"]
        t_cfg = config["trading"]
        rt_cfg = config["runtime"]

        self.api = PredictApi(
            base_url=api_cfg["base_url"],
            api_key=api_cfg["api_key"],
            timeout=int(api_cfg.get("timeout_seconds", 20)),
        )

        self.private_key = wallet_cfg["private_key"]
        validate_private_key(self.private_key)
        self.predict_account = (wallet_cfg.get("predict_account") or "").strip() or None
        self.account = Account.from_key(self.private_key)

        self.buy_price = parse_decimal(t_cfg["buy_price_per_share"], "trading.buy_price_per_share")
        self.buy_shares = parse_decimal(t_cfg["buy_shares"], "trading.buy_shares")
        self.post_only_required = bool(t_cfg.get("post_only_required", True))

        self.market_prefix = str(t_cfg.get("market_title_prefix", "BTC/USD Up or Down"))
        self.duration_min = int(t_cfg.get("market_duration_minutes", 15))
        self.poll_interval = int(t_cfg.get("poll_interval_seconds", 2))
        self.market_fetch_size = int(t_cfg.get("market_fetch_size", 50))
        self.chain_id_raw = t_cfg.get("chain_id", 56)

        self.dry_run = bool(rt_cfg.get("dry_run", False))
        self.status_log_interval = int(rt_cfg.get("status_log_interval_seconds", 2))
        self._last_status_log_at: Optional[datetime] = None

        self.current_price_provider = str(rt_cfg.get("current_price_provider", "binance")).lower()
        self.current_price_ttl_seconds = int(rt_cfg.get("current_price_ttl_seconds", 2))
        self.binance_symbol = str(rt_cfg.get("binance_symbol", "BTCUSDT")).upper()

        self.market_guard_seconds = int(rt_cfg.get("market_guard_seconds", 30))
        # Keep legacy key name for compatibility.
        # Semantics: ratio threshold (e.g. 0.0008 = 0.08%)
        self.gap_keep_threshold = parse_decimal(rt_cfg.get("gap_keep_threshold_usd", "0.0008"), "runtime.gap_keep_threshold_usd")
        self.gap_check_start_seconds = int(rt_cfg.get("gap_check_start_seconds", self.market_guard_seconds))
        self.force_cancel_seconds = int(rt_cfg.get("force_cancel_seconds", 10))
        self.cancel_retry_interval_seconds = int(rt_cfg.get("cancel_retry_interval_seconds", 5))

        if self.gap_keep_threshold < 0:
            raise ValueError("runtime.gap_keep_threshold_usd must be >= 0 (ratio, e.g. 0.0008 = 0.08%)")

        self._cached_current_price: Optional[str] = None
        self._cached_current_price_at: Optional[datetime] = None
        self._last_skip_market_id: Optional[str] = None

        self.ob = self._build_order_builder()
        self.active_cycle: Optional[MarketCycle] = None

    def _build_order_builder(self) -> OrderBuilder:
        opts = OrderBuilderOptions(predict_account=self.predict_account) if self.predict_account else OrderBuilderOptions()
        attempts = []
        signature_text = ""
        try:
            signature_text = str(inspect.signature(OrderBuilder.make))
        except Exception:
            signature_text = ""

        uses_chain_id = "chain_id" in signature_text
        chain_candidates: List[Any] = [self.chain_id_raw]
        try:
            chain_candidates.append(int(str(self.chain_id_raw)))
        except Exception:
            pass

        try:
            from predict_sdk.types import ChainId as SDKChainId  # type: ignore

            for key in [str(self.chain_id_raw).upper(), "BSC", "BSC_MAINNET", "MAINNET"]:
                if hasattr(SDKChainId, key):
                    chain_candidates.append(getattr(SDKChainId, key))
            for value in SDKChainId:
                chain_candidates.append(value)
        except Exception:
            pass

        if uses_chain_id:
            for c in chain_candidates:
                attempts.append(lambda c=c: OrderBuilder.make(chain_id=c, signer=self.private_key, options=opts))
                attempts.append(lambda c=c: OrderBuilder.make(c, self.private_key, opts))
                attempts.append(lambda c=c: OrderBuilder.make(c, self.private_key))
        else:
            attempts.append(lambda: OrderBuilder.make(private_key=self.private_key, options=opts))
            attempts.append(lambda: OrderBuilder.make(private_key=self.private_key))
            attempts.append(lambda: OrderBuilder.make(wallet_private_key=self.private_key, options=opts))
            attempts.append(lambda: OrderBuilder.make(wallet_private_key=self.private_key))
            if self.predict_account:
                attempts.append(lambda: OrderBuilder.make(private_key=self.private_key, predict_account=self.predict_account))
                attempts.append(lambda: OrderBuilder.make(wallet_private_key=self.private_key, predict_account=self.predict_account))
            attempts.append(lambda: OrderBuilder.make(self.private_key, opts))
            attempts.append(lambda: OrderBuilder.make(self.private_key))

        attempts.append(lambda: OrderBuilder(self.private_key, opts))
        attempts.append(lambda: OrderBuilder(self.private_key))

        errors: List[str] = []
        for fn in attempts:
            try:
                ob = fn()
                if ob:
                    logging.info("🧩 OrderBuilder 初始化成功")
                    return ob
            except Exception as exc:
                errors.append(str(exc))

        sig = signature_text or "<unknown>"
        raise RuntimeError(
            "failed to initialize OrderBuilder for installed predict-sdk. "
            f"OrderBuilder.make signature={sig}. errors={'; '.join(errors[:6])}"
        )

    def _build_auth_signature(self, message: str) -> str:
        if self.predict_account:
            return self.ob.sign_predict_account_message(message)
        sig = Account.sign_message(encode_defunct(text=message), private_key=self.private_key)
        return sig.signature.hex()

    def auth(self) -> None:
        signer = self.predict_account or self.account.address
        msg_resp = self.api._request("GET", "/v1/auth/message", auth=False)
        message = msg_resp.get("data", {}).get("message") or msg_resp.get("message")
        if not message:
            raise RuntimeError("auth message missing")

        signature = self._build_auth_signature(message)
        payload = {"signer": signer, "message": message, "signature": signature}
        auth_resp = self.api._request("POST", "/v1/auth", data=payload, auth=False)
        token = (
            auth_resp.get("data", {}).get("accessToken")
            or auth_resp.get("data", {}).get("token")
            or auth_resp.get("accessToken")
            or auth_resp.get("token")
        )
        if not token:
            raise RuntimeError("accessToken missing")
        self.api.jwt_token = token
        logging.info("🔐 鉴权成功 signer=%s", signer)

    def maybe_set_approvals(self) -> None:
        if not bool(self.cfg["trading"].get("auto_set_approvals", False)):
            return
        if self.dry_run:
            logging.info("dry_run=true，跳过 approvals 设置")
            return
        logging.info("⛓️ 开始链上设置 approvals")
        self.ob.set_approvals()
        logging.info("✅ approvals 设置完成")

    def pick_latest_open_market(self) -> Optional[Dict[str, Any]]:
        markets = self.api.get_markets(first=self.market_fetch_size, status="OPEN")
        filtered = [m for m in markets if get_market_title(m).startswith(self.market_prefix)]
        if not filtered:
            return None
        filtered.sort(key=market_sort_key, reverse=True)
        return filtered[0]

    def _seconds_to_market_end(self, market: Dict[str, Any]) -> Optional[int]:
        _, et = parse_market_times(market, self.duration_min)
        if et is None:
            return None
        return int((et - now_utc()).total_seconds())

    def should_enter_market_now(self, market: Dict[str, Any]) -> bool:
        end_secs = self._seconds_to_market_end(market)
        if end_secs is None:
            return True
        return end_secs > self.market_guard_seconds

    def _merge_market_with_detail(self, cycle: MarketCycle, market: Dict[str, Any]) -> Dict[str, Any]:
        ptb, cp = extract_market_prices(market)
        if ptb != "n/a" or cp != "n/a":
            return market

        now = now_utc()
        need_fetch = cycle.market_detail_fetched_at is None or (now - cycle.market_detail_fetched_at).total_seconds() >= 10
        if need_fetch:
            try:
                detail = self.api.get_market(cycle.market_id)
                if isinstance(detail, dict) and detail:
                    cycle.market_detail = detail
                cycle.market_detail_fetched_at = now
            except Exception as exc:
                logging.warning("拉取市场详情失败 market=%s err=%s", cycle.market_id, exc)
                cycle.market_detail_fetched_at = now

        merged = dict(market)
        if cycle.market_detail:
            merged.update(cycle.market_detail)
        return merged

    def _fetch_external_current_price(self) -> str:
        if self.current_price_provider != "binance":
            return "n/a"

        now = now_utc()
        if (
            self._cached_current_price is not None
            and self._cached_current_price_at is not None
            and (now - self._cached_current_price_at).total_seconds() < self.current_price_ttl_seconds
        ):
            return self._cached_current_price

        try:
            resp = requests.get(
                "https://api.binance.com/api/v3/ticker/price",
                params={"symbol": self.binance_symbol},
                timeout=self.api.timeout,
            )
            if resp.status_code >= 400:
                return "n/a"
            data = resp.json()
            p = data.get("price")
            if p is None:
                return "n/a"
            self._cached_current_price = str(p)
            self._cached_current_price_at = now
            return self._cached_current_price
        except Exception:
            return "n/a"

    def _fetch_binance_price_at_start(self, start_at: datetime) -> Tuple[Optional[str], Optional[datetime]]:
        if self.current_price_provider != "binance":
            return None, None
        try:
            start_ms = int(start_at.timestamp() * 1000)
            end_ms = start_ms + 60_000
            resp = requests.get(
                "https://api.binance.com/api/v3/klines",
                params={
                    "symbol": self.binance_symbol,
                    "interval": "1m",
                    "startTime": start_ms,
                    "endTime": end_ms,
                    "limit": 1,
                },
                timeout=self.api.timeout,
            )
            if resp.status_code >= 400:
                return None, None
            data = resp.json()
            if not isinstance(data, list) or not data:
                return None, None
            candle = data[0]
            if not isinstance(candle, list) or len(candle) < 2:
                return None, None
            candle_open_time = datetime.fromtimestamp(int(candle[0]) / 1000, tz=timezone.utc)
            candle_open_price = str(candle[1])
            return candle_open_price, candle_open_time
        except Exception:
            return None, None

    def _compute_gap(self, cycle: MarketCycle, market: Dict[str, Any]) -> Tuple[Optional[Decimal], Optional[Decimal], Dict[str, str]]:
        market = self._merge_market_with_detail(cycle, market)
        ptb_now, market_current = extract_market_prices(market)
        current_price = self._fetch_external_current_price() if market_current == "n/a" else market_current

        metrics = {
            "ptb_now": ptb_now,
            "current_price": current_price,
            "mapped_ptb": "n/a",
            "gap": "n/a",
            "gap_ratio": "n/a",
        }

        if cycle.ptb_binance_offset is None or ptb_now == "n/a" or current_price == "n/a":
            return None, None, metrics

        try:
            offset = Decimal(cycle.ptb_binance_offset)
            mapped_ptb = Decimal(ptb_now) - offset
            gap = abs(Decimal(current_price) - mapped_ptb)
            denom = abs(mapped_ptb)
            gap_ratio: Optional[Decimal] = None
            if denom != 0:
                gap_ratio = gap / denom
            metrics["mapped_ptb"] = f"{mapped_ptb:.6f}"
            metrics["gap"] = f"{gap:.6f}"
            if gap_ratio is not None:
                metrics["gap_ratio"] = f"{gap_ratio:.8f}"
            return gap, gap_ratio, metrics
        except Exception:
            return None, None, metrics

    def _find_remote_order(
        self,
        lo: LocalOrder,
        remote_by_hash: Dict[str, Dict[str, Any]],
        remote_by_id: Dict[str, Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if lo.order_id:
            ro = remote_by_id.get(str(lo.order_id))
            if ro is not None:
                return ro
        return remote_by_hash.get(str(lo.order_hash))

    def _normalize_any_qty_to_wei(self, raw: Any) -> int:
        if raw is None:
            return 0
        try:
            d = Decimal(str(raw))
        except (InvalidOperation, ValueError):
            return 0
        if d <= 0:
            return 0
        if d >= Decimal("1000000000000"):
            return int(d)
        return to_wei(d)

    def _extract_filled_wei(self, ro: Optional[Dict[str, Any]], fallback: int = 0, total_qty_wei: Optional[int] = None) -> int:
        if not ro:
            return fallback

        candidates: List[int] = []
        for k in [
            "amountFilled",
            "amount_filled",
            "filledAmount",
            "filled_amount",
            "filled",
            "filledShares",
            "filled_shares",
            "executedAmount",
            "executed_amount",
            "matchedAmount",
            "matched_amount",
            "sizeFilled",
            "size_filled",
        ]:
            if k in ro and ro[k] is not None:
                candidates.append(self._normalize_any_qty_to_wei(ro[k]))

        if total_qty_wei is not None:
            rem = maybe_get(ro, "amountRemaining", "amount_remaining", "remainingAmount", "remaining_amount", "remaining")
            if rem is not None:
                rem_wei = self._normalize_any_qty_to_wei(rem)
                if rem_wei <= total_qty_wei:
                    candidates.append(total_qty_wei - rem_wei)

        v = max(candidates) if candidates else 0
        return max(v, fallback)

    def _extract_status(self, ro: Optional[Dict[str, Any]], fallback: str = "UNKNOWN") -> str:
        if not ro:
            return fallback
        return str(maybe_get(ro, "status") or fallback).upper()

    def _refresh_tracked_orders_by_id(
        self,
        cycle: MarketCycle,
        remote_by_hash: Dict[str, Dict[str, Any]],
        remote_by_id: Dict[str, Dict[str, Any]],
    ) -> None:
        for lo in cycle.buy_orders:
            if not lo.order_id:
                continue
            oid = str(lo.order_id)
            if oid in remote_by_id:
                continue
            try:
                ro = self.api.get_order(oid)
                if not ro:
                    continue
                remote_by_id[oid] = ro
                oh = maybe_get(ro, "hash", "orderHash", "order_hash")
                if oh is not None:
                    remote_by_hash[str(oh)] = ro
            except Exception:
                pass

    def _submit_cancel_request(self, ids: List[str]) -> Tuple[int, List[str]]:
        ids = [str(i) for i in ids if i]
        if not ids:
            return 0, []

        if self.dry_run:
            logging.info("dry_run=true，跳过撤单 ids=%s", ",".join(ids))
            return len(ids), ids

        try:
            resp = self.api.remove_orders(ids)
            data = resp.get("data", {}) if isinstance(resp, dict) else {}
            removed = data.get("removed", []) if isinstance(data, dict) else []
            noop = data.get("noop", []) if isinstance(data, dict) else []
            handled_ids = [str(i) for i in list(removed) + list(noop) if i is not None]
            if not handled_ids:
                handled_ids = ids
            return len(ids), handled_ids
        except Exception as exc:
            logging.warning("撤单请求失败 ids=%s err=%s", ",".join(ids), exc)
            return 0, []

    def _mark_cancel_handled_for_orders(self, orders: List[LocalOrder], handled_ids: List[str]) -> None:
        if not handled_ids:
            return
        handled = {str(i) for i in handled_ids}
        for lo in orders:
            if lo.order_id and str(lo.order_id) in handled:
                lo.cancel_finalized = True
                if lo.last_known_filled_wei < lo.quantity_wei:
                    lo.last_known_status = "CANCELLED_LOCAL"

    def _cancel_unfilled_buy_orders(
        self,
        cycle: MarketCycle,
        remote_by_hash: Dict[str, Dict[str, Any]],
        remote_by_id: Dict[str, Dict[str, Any]],
        reason: str,
    ) -> int:
        ids_to_cancel: List[str] = []
        to_mark: List[LocalOrder] = []

        for lo in cycle.buy_orders:
            ro = self._find_remote_order(lo, remote_by_hash, remote_by_id)
            filled_wei = self._extract_filled_wei(ro, fallback=lo.last_known_filled_wei, total_qty_wei=lo.quantity_wei)
            if filled_wei >= lo.quantity_wei:
                continue
            if lo.cancel_requested and lo.cancel_finalized:
                continue
            oid = maybe_get(ro or {}, "id", "orderId", "order_id") or lo.order_id
            if oid:
                ids_to_cancel.append(str(oid))
                to_mark.append(lo)

        ids_to_cancel = list(dict.fromkeys(ids_to_cancel))
        if not ids_to_cancel:
            return 0

        now = now_utc()
        for lo in to_mark:
            lo.cancel_requested = True
            lo.cancel_attempts += 1
            lo.last_cancel_attempt_at = now

        logging.info("🛑 %s，撤单数量=%d ids=%s", reason, len(ids_to_cancel), ",".join(ids_to_cancel))
        _, handled_ids = self._submit_cancel_request(ids_to_cancel)
        self._mark_cancel_handled_for_orders(to_mark, handled_ids)
        return len(ids_to_cancel)

    def _retry_pending_buy_cancels(
        self,
        cycle: MarketCycle,
        remote_by_hash: Dict[str, Dict[str, Any]],
        remote_by_id: Dict[str, Dict[str, Any]],
    ) -> None:
        ids_to_cancel: List[str] = []
        to_mark: List[LocalOrder] = []
        now = now_utc()

        for lo in cycle.buy_orders:
            if not lo.cancel_requested or not lo.order_id or lo.cancel_finalized:
                continue
            if lo.last_cancel_attempt_at and (now - lo.last_cancel_attempt_at).total_seconds() < self.cancel_retry_interval_seconds:
                continue

            ro = self._find_remote_order(lo, remote_by_hash, remote_by_id)
            if ro is None:
                lo.cancel_missing_polls += 1
                if lo.cancel_missing_polls >= 2:
                    lo.cancel_finalized = True
                    lo.last_known_status = "CANCELLED_LOCAL"
                continue

            lo.cancel_missing_polls = 0
            status = self._extract_status(ro, fallback=lo.last_known_status)
            filled_wei = self._extract_filled_wei(ro, fallback=lo.last_known_filled_wei, total_qty_wei=lo.quantity_wei)
            if status in {"FILLED", "CANCELLED", "CANCELED", "CLOSED", "EXPIRED", "CANCELLED_LOCAL"}:
                continue
            if filled_wei >= lo.quantity_wei:
                continue

            ids_to_cancel.append(str(lo.order_id))
            to_mark.append(lo)

        ids_to_cancel = list(dict.fromkeys(ids_to_cancel))
        if not ids_to_cancel:
            return

        for lo in to_mark:
            lo.cancel_attempts += 1
            lo.last_cancel_attempt_at = now

        logging.info("🔁 撤单重试 pending_count=%d ids=%s", len(ids_to_cancel), ",".join(ids_to_cancel))
        _, handled_ids = self._submit_cancel_request(ids_to_cancel)
        self._mark_cancel_handled_for_orders(to_mark, handled_ids)

    def cancel_opposite_buy_if_filled(
        self,
        cycle: MarketCycle,
        remote_by_hash: Dict[str, Dict[str, Any]],
        remote_by_id: Dict[str, Dict[str, Any]],
    ) -> None:
        filled_outcomes = set()
        buy_filled: Dict[str, int] = {}

        for lo in cycle.buy_orders:
            ro = self._find_remote_order(lo, remote_by_hash, remote_by_id)
            filled_wei = self._extract_filled_wei(ro, fallback=lo.last_known_filled_wei, total_qty_wei=lo.quantity_wei)
            buy_filled[lo.order_hash] = filled_wei
            if filled_wei > 0:
                filled_outcomes.add(lo.outcome)

        if not filled_outcomes:
            return

        ids_to_cancel: List[str] = []
        to_mark: List[LocalOrder] = []

        for lo in cycle.buy_orders:
            if lo.outcome in filled_outcomes:
                continue
            if lo.cancel_requested:
                continue
            if buy_filled.get(lo.order_hash, 0) >= lo.quantity_wei:
                continue

            ro = self._find_remote_order(lo, remote_by_hash, remote_by_id)
            oid = maybe_get(ro or {}, "id", "orderId", "order_id") or lo.order_id
            if oid:
                ids_to_cancel.append(str(oid))
                to_mark.append(lo)

        ids_to_cancel = list(dict.fromkeys(ids_to_cancel))
        if not ids_to_cancel:
            return

        now = now_utc()
        for lo in to_mark:
            lo.cancel_requested = True
            lo.cancel_attempts += 1
            lo.last_cancel_attempt_at = now

        logging.info("⚖️ 单边成交，撤销对手方向买单 count=%d", len(ids_to_cancel))
        _, handled_ids = self._submit_cancel_request(ids_to_cancel)
        self._mark_cancel_handled_for_orders(to_mark, handled_ids)

    def _should_log_status(self) -> bool:
        if self.status_log_interval <= 0:
            return False
        now = now_utc()
        if self._last_status_log_at is None:
            self._last_status_log_at = now
            return True
        if (now - self._last_status_log_at).total_seconds() >= self.status_log_interval:
            self._last_status_log_at = now
            return True
        return False

    def _seconds_to_text(self, target: Optional[datetime]) -> str:
        if not target:
            return "n/a"
        secs = int((target - now_utc()).total_seconds())
        return self._seconds_value_to_text(secs)

    def _seconds_value_to_text(self, secs: int) -> str:
        sign = "-" if secs < 0 else ""
        secs = abs(secs)
        mm, ss = divmod(secs, 60)
        hh, mm = divmod(mm, 60)
        return f"{sign}{hh:02d}:{mm:02d}:{ss:02d}"

    def log_cycle_status(
        self,
        cycle: MarketCycle,
        market: Dict[str, Any],
        remote_by_hash: Dict[str, Dict[str, Any]],
        remote_by_id: Dict[str, Dict[str, Any]],
        gap: Optional[Decimal],
        gap_ratio: Optional[Decimal],
        gap_reason: str,
    ) -> None:
        if not self._should_log_status():
            return

        ptb_now, _ = extract_market_prices(self._merge_market_with_detail(cycle, market))
        current_price = self._fetch_external_current_price()
        mapped_ptb = "n/a"
        if cycle.ptb_binance_offset is not None and ptb_now != "n/a":
            try:
                mapped_ptb = f"{(Decimal(ptb_now) - Decimal(cycle.ptb_binance_offset)):.6f}"
            except Exception:
                mapped_ptb = "n/a"

        end_secs = int((cycle.end_at - now_utc()).total_seconds()) if cycle.end_at else None
        gap_check_left_text = "n/a"
        if end_secs is not None:
            gap_check_left_text = self._seconds_value_to_text(end_secs - self.gap_check_start_seconds)

        threshold_text = str(self.gap_keep_threshold)
        threshold_pct_text = f"{(self.gap_keep_threshold * Decimal('100')):.4f}%"
        if gap_ratio is None:
            gap_compare_text = "n/a"
        elif gap_ratio <= self.gap_keep_threshold:
            gap_compare_text = (
                f"{gap_ratio:.8f} ({(gap_ratio * Decimal('100')):.4f}%) "
                f"<= {threshold_text} ({threshold_pct_text}) (保留)"
            )
        else:
            gap_compare_text = (
                f"{gap_ratio:.8f} ({(gap_ratio * Decimal('100')):.4f}%) "
                f"> {threshold_text} ({threshold_pct_text}) (撤单)"
            )

        lines = [
            f"[状态] market={cycle.market_id} slug={cycle.market_slug}",
            f"[状态] 距离结束={self._seconds_to_text(cycle.end_at)} 距离开始价差检查={gap_check_left_text} "
            f"规则判定={gap_reason} gap(USD)={gap if gap is not None else 'n/a'} "
            f"gap_ratio={gap_ratio if gap_ratio is not None else 'n/a'}",
            f"[状态] gap阈值(比例)={threshold_text} ({threshold_pct_text}) gap对比={gap_compare_text}",
            f"[状态] binance_open={cycle.binance_open or 'n/a'} ptb_open={cycle.ptb_open or 'n/a'} offset={cycle.ptb_binance_offset or 'n/a'}",
            f"[状态] ptb_now={ptb_now} mapped_ptb={mapped_ptb} current_price(binance)={current_price}",
        ]

        for lo in cycle.buy_orders:
            ro = self._find_remote_order(lo, remote_by_hash, remote_by_id)
            status = self._extract_status(ro, fallback=lo.last_known_status)
            filled_wei = self._extract_filled_wei(ro, fallback=lo.last_known_filled_wei, total_qty_wei=lo.quantity_wei)
            if lo.cancel_finalized and filled_wei < lo.quantity_wei:
                status = "CANCELLED_LOCAL"
            lines.append(
                "[状态] 买单 "
                f"方向={lo.outcome} order_id={lo.order_id or '-'} 状态={status} "
                f"成交={Decimal(filled_wei)/WAD}/{Decimal(lo.quantity_wei)/WAD} "
                f"已发起撤单={lo.cancel_requested} 撤单尝试次数={lo.cancel_attempts}"
            )

        for line in lines:
            logging.info(line)

    def build_order_payload(
        self,
        *,
        market: Dict[str, Any],
        token_id: str,
        side: Side,
        price_per_share: Decimal,
        quantity_wei: int,
    ) -> Tuple[Dict[str, Any], str]:
        price_wei = to_wei(price_per_share)
        amounts = self.ob.get_limit_order_amounts(
            LimitHelperInput(side=side, price_per_share_wei=price_wei, quantity_wei=quantity_wei)
        )

        fee_rate_bps = int(maybe_get(market, "feeRateBps", "fee_rate_bps") or 0)
        order = self.ob.build_order(
            "LIMIT",
            BuildOrderInput(
                side=side,
                token_id=token_id,
                maker_amount=str(amounts.maker_amount),
                taker_amount=str(amounts.taker_amount),
                fee_rate_bps=fee_rate_bps,
            ),
        )

        typed = self.ob.build_typed_data(
            order,
            is_neg_risk=bool(maybe_get(market, "isNegRisk", "is_neg_risk") or False),
            is_yield_bearing=bool(maybe_get(market, "isYieldBearing", "is_yield_bearing") or False),
        )
        signed = self.ob.sign_typed_data_order(typed)
        order_hash = self.ob.build_typed_data_hash(typed)
        signed_order = normalize_signed_order(signed, order_hash)

        payload = {
            "data": {
                "strategy": "LIMIT",
                "pricePerShare": str(price_wei),
                "side": int(side),
                "tokenId": str(token_id),
                "isFillOrKill": False,
                "postOnly": True,
                "isPostOnly": True,
                "slippageBps": "0",
                "order": signed_order,
            }
        }
        return payload, order_hash

    def place_limit_order(
        self,
        *,
        market: Dict[str, Any],
        outcome: str,
        token_id: str,
        side: Side,
        price_per_share: Decimal,
        quantity_wei: int,
    ) -> LocalOrder:
        payload, order_hash = self.build_order_payload(
            market=market,
            token_id=token_id,
            side=side,
            price_per_share=price_per_share,
            quantity_wei=quantity_wei,
        )

        if self.dry_run:
            remote_order_id = None
            remote_order_hash = None
            initial_status = "UNKNOWN"
            initial_filled_wei = 0
            logging.info("dry_run=true，跳过下单 side=%s outcome=%s", "BUY" if side == Side.BUY else "SELL", outcome)
        else:
            try:
                create_resp = self.api.create_order(payload)
            except Exception as exc:
                s = str(exc).lower()
                if self.post_only_required and any(k in s for k in POST_ONLY_ERROR_HINTS):
                    raise RuntimeError(f"post-only order rejected and strict mode enabled: {exc}") from exc
                raise

            remote_order_id, remote_order_hash = extract_order_id_and_hash(create_resp)
            initial_status, initial_filled_wei = extract_order_status_and_filled_wei(create_resp)
            if remote_order_hash:
                order_hash = remote_order_hash

        lo = LocalOrder(
            outcome=outcome,
            token_id=token_id,
            side="BUY" if side == Side.BUY else "SELL",
            price_per_share=price_per_share,
            quantity_wei=quantity_wei,
            order_hash=order_hash,
            order_id=remote_order_id,
        )
        lo.last_known_status = initial_status
        lo.last_known_filled_wei = initial_filled_wei
        return lo

    def start_cycle(self, market: Dict[str, Any]) -> MarketCycle:
        market_id = str(maybe_get(market, "id", ""))
        title = get_market_title(market)
        slug = str(maybe_get(market, "slug", ""))
        st, et = parse_market_times(market, self.duration_min)

        up_token, down_token = pick_up_down_tokens(market)
        qty_wei = to_wei(self.buy_shares)

        up_buy = self.place_limit_order(
            market=market,
            outcome="UP",
            token_id=up_token,
            side=Side.BUY,
            price_per_share=self.buy_price,
            quantity_wei=qty_wei,
        )
        down_buy = self.place_limit_order(
            market=market,
            outcome="DOWN",
            token_id=down_token,
            side=Side.BUY,
            price_per_share=self.buy_price,
            quantity_wei=qty_wei,
        )

        cycle = MarketCycle(
            market_id=market_id,
            market_title=title,
            market_slug=slug,
            start_at=st,
            end_at=et,
            buy_orders=[up_buy, down_buy],
        )

        market = self._merge_market_with_detail(cycle, market)
        ptb_open, _ = extract_market_prices(market)
        cycle.ptb_open = None if ptb_open == "n/a" else ptb_open

        if st is not None:
            b0, b0_ts = self._fetch_binance_price_at_start(st)
            cycle.binance_open = b0
            cycle.binance_open_time = b0_ts

        if cycle.ptb_open is not None and cycle.binance_open is not None:
            try:
                offset = Decimal(cycle.ptb_open) - Decimal(cycle.binance_open)
                cycle.ptb_binance_offset = str(offset)
            except Exception:
                cycle.ptb_binance_offset = None

        logging.info(
            "🚀 新周期开始 market=%s start=%s end=%s binance_open=%s ptb_open=%s offset=%s",
            cycle.market_id,
            cycle.start_at,
            cycle.end_at,
            cycle.binance_open,
            cycle.ptb_open,
            cycle.ptb_binance_offset,
        )
        return cycle

    def monitor_cycle(self, cycle: MarketCycle, market: Dict[str, Any]) -> None:
        orders = self.api.get_orders(first=300)
        remote_by_hash: Dict[str, Dict[str, Any]] = {}
        remote_by_id: Dict[str, Dict[str, Any]] = {}

        for o in orders:
            h = str(maybe_get(o, "hash", "orderHash", "order_hash") or "")
            if h:
                remote_by_hash[h] = o
            oid = maybe_get(o, "id", "orderId", "order_id")
            if oid is not None:
                remote_by_id[str(oid)] = o

        self._refresh_tracked_orders_by_id(cycle, remote_by_hash, remote_by_id)

        for lo in cycle.buy_orders:
            ro = self._find_remote_order(lo, remote_by_hash, remote_by_id)
            if not ro:
                continue
            lo.last_known_status = self._extract_status(ro, fallback=lo.last_known_status)
            lo.last_known_filled_wei = self._extract_filled_wei(ro, fallback=lo.last_known_filled_wei, total_qty_wei=lo.quantity_wei)

        self.cancel_opposite_buy_if_filled(cycle, remote_by_hash, remote_by_id)

        end_secs = int((cycle.end_at - now_utc()).total_seconds()) if cycle.end_at else None
        gap, gap_ratio, _metrics = self._compute_gap(cycle, market)
        reason, should_cancel = evaluate_cancel_decision(
            end_secs,
            gap_ratio,
            force_cancel_seconds=self.force_cancel_seconds,
            gap_keep_threshold=self.gap_keep_threshold,
            gap_check_start_seconds=self.gap_check_start_seconds,
        )

        if should_cancel:
            canceled = self._cancel_unfilled_buy_orders(cycle, remote_by_hash, remote_by_id, f"触发撤单规则: {reason}")
            if canceled > 0:
                cycle.gap_rule_cancelled = True

        self._retry_pending_buy_cancels(cycle, remote_by_hash, remote_by_id)
        self.log_cycle_status(cycle, market, remote_by_hash, remote_by_id, gap, gap_ratio, reason)

    def loop(self) -> None:
        self.auth()
        self.maybe_set_approvals()

        while True:
            try:
                market = self.pick_latest_open_market()
                if not market:
                    logging.info("当前无匹配开放市场 prefix=%s", self.market_prefix)
                    time.sleep(self.poll_interval)
                    continue

                current_market_id = str(maybe_get(market, "id", ""))
                if not current_market_id:
                    logging.warning("匹配到的市场缺少 id，跳过")
                    time.sleep(self.poll_interval)
                    continue

                if self.active_cycle is None or self.active_cycle.market_id != current_market_id:
                    if not self.should_enter_market_now(market):
                        if self._last_skip_market_id != current_market_id:
                            st, et = parse_market_times(market, self.duration_min)
                            end_secs = self._seconds_to_market_end(market)
                            logging.info(
                                "⏱️ 市场已进入保护窗口，跳过下单 market=%s start=%s end=%s end_secs=%s guard=%s",
                                current_market_id,
                                st,
                                et,
                                end_secs,
                                self.market_guard_seconds,
                            )
                            self._last_skip_market_id = current_market_id
                        time.sleep(self.poll_interval)
                        continue

                    self._last_skip_market_id = None
                    self.active_cycle = self.start_cycle(market)

                self.monitor_cycle(self.active_cycle, market)
                time.sleep(self.poll_interval)
            except RuntimeError as exc:
                if "401" in str(exc) or "403" in str(exc):
                    logging.warning("⚠️ 鉴权可能过期，尝试重新鉴权: %s", exc)
                    self.auth()
                else:
                    logging.exception("❌ 运行时错误: %s", exc)
                    time.sleep(self.poll_interval)
            except Exception:
                logging.exception("❌ 未预期异常")
                time.sleep(self.poll_interval)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("config root must be object")
    return cfg


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Predict.fun BTC/USD Up or Down auto trader")
    p.add_argument("--config", default="config.yaml", help="config yaml path")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)

    runtime_cfg = cfg.get("runtime", {})
    log_level = str(runtime_cfg.get("log_level", "INFO")).upper()
    log_format = "%(asctime)s | %(levelname)s | %(message)s"
    handlers: List[logging.Handler] = [logging.StreamHandler()]

    log_file = str(runtime_cfg.get("log_file", "") or "").strip()
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format=log_format, handlers=handlers)

    trader = PredictTrader(cfg)
    trader.loop()


if __name__ == "__main__":
    main()
