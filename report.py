"""
定时研究报告模块 — 时段转换时生成 Top N 币种综合研究报告并发送邮箱。

功能:
  - 从 CoinResearchCache 提取 Top N 币种完整分析
  - 生成 HTML 格式研究报告（含表格、信号、体制分布）
  - 通过 SMTP 发送到指定邮箱
  - 报告存档到本地 reports/ 目录
"""

import logging
import os
import smtplib
import threading
import time
import locale
from datetime import datetime, timezone, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import Optional

logger = logging.getLogger(__name__)

# 北京时间时区
BEIJING_TZ = timezone(timedelta(hours=8))


# ---------------------------------------------------------------------------
# 时段检测
# ---------------------------------------------------------------------------

class SessionReporter:
    """时段报告管理器 — 检测是否到达时段转换点，避免重复触发。"""

    def __init__(self, config: dict):
        report_cfg = config.get("report", {})
        self.enabled = report_cfg.get("enabled", True)
        self.session_times = report_cfg.get("session_times", ["08:00", "15:00", "20:00"])
        self.top_n = report_cfg.get("top_n_display", 30)
        self.report_dir = report_cfg.get("report_dir", "reports")
        self.email_cfg = report_cfg.get("email", {})

        # 追踪已触发的时段（防止重复触发）
        self._lock = threading.Lock()
        self._triggered_minutes: set = set()
        self._last_report_timestamp = 0.0

    def should_generate_report(self) -> bool:
        """检查是否到了时段转换点且尚未触发。"""
        if not self.enabled:
            return False

        now = datetime.now(BEIJING_TZ)
        current_time = now.strftime("%H:%M")
        current_ts = datetime.now(timezone.utc).timestamp()

        with self._lock:
            # 每天重置（基于 UTC 时间戳判断是否跨天）
            if current_ts - self._last_report_timestamp > 86400:
                self._triggered_minutes.clear()
                self._last_report_timestamp = current_ts

            # 检查是否匹配任一时段
            for session_time in self.session_times:
                if current_time == session_time and session_time not in self._triggered_minutes:
                    self._triggered_minutes.add(session_time)
                    return True

        return False

    def get_session_label(self) -> str:
        """获取当前时段标签。"""
        now = datetime.now(BEIJING_TZ)

        session_names = {
            "08:00": "亚洲开盘",
            "15:00": "欧洲开盘",
            "20:00": "美国开盘",
        }
        current_time = now.strftime("%H:%M")
        return session_names.get(current_time, "时段报告")


# ---------------------------------------------------------------------------
# HTML 报告生成
# ---------------------------------------------------------------------------

def generate_html_report(
    cache_entries: list,
    session_label: str,
    config: dict,
) -> str:
    """生成 HTML 格式的研究报告。

    Args:
        cache_entries: CoinResearchEntry 列表
        session_label: 时段标签（如 "亚洲开盘"）
        config: 完整配置

    Returns:
        HTML 字符串
    """
    report_cfg = config.get("report", {})
    top_n = report_cfg.get("top_n_display", 30)
    now = datetime.now(BEIJING_TZ)

    # 筛选有分析的条目，按置信度降序
    analyzed = sorted(
        [e for e in cache_entries if e.has_analysis and e.analysis is not None],
        key=lambda e: e.analysis.final_confidence if e.analysis else 0,
        reverse=True,
    )[:top_n]

    # 统计
    total = len(cache_entries)
    with_signals = sum(1 for e in cache_entries if e.has_signals)
    with_analysis = sum(1 for e in cache_entries if e.has_analysis)
    regime_counts = {"FUSED": 0, "DIVERGENT": 0, "NEUTRAL": 0, "TRANSITIONING": 0}
    for e in cache_entries:
        regime = e.detected_regime or "NEUTRAL"
        if regime in regime_counts:
            regime_counts[regime] += 1

    # 构建 HTML
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OKX FVG 研究报告 — {session_label}</title>
<style>
  body {{ font-family: -apple-system, 'Microsoft YaHei', sans-serif; background: #0d1117; color: #c9d1d9; margin: 0; padding: 20px; }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  h1 {{ color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 10px; }}
  h2 {{ color: #f0f6fc; margin-top: 30px; }}
  .summary {{ display: flex; gap: 15px; flex-wrap: wrap; margin: 20px 0; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px 20px; flex: 1; min-width: 120px; text-align: center; }}
  .card .value {{ font-size: 28px; font-weight: bold; color: #58a6ff; }}
  .card .label {{ font-size: 12px; color: #8b949e; margin-top: 5px; }}
  .card.green .value {{ color: #3fb950; }}
  .card.red .value {{ color: #f85149; }}
  table {{ width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 13px; }}
  th {{ background: #161b22; color: #8b949e; text-align: left; padding: 10px 8px; border-bottom: 2px solid #30363d; white-space: nowrap; }}
  td {{ padding: 8px; border-bottom: 1px solid #21262d; }}
  tr:hover {{ background: #1c2128; }}
  .long {{ color: #3fb950; }}
  .short {{ color: #f85149; }}
  .score-positive {{ color: #3fb950; }}
  .score-negative {{ color: #f85149; }}
  .regime {{ font-size: 11px; padding: 2px 6px; border-radius: 4px; }}
  .regime-FUSED {{ background: #1b3d1b; color: #3fb950; }}
  .regime-DIVERGENT {{ background: #3d1b1b; color: #f85149; }}
  .regime-NEUTRAL {{ background: #1b2b3d; color: #58a6ff; }}
  .regime-TRANSITIONING {{ background: #3d3d1b; color: #d2991d; }}
  .footer {{ margin-top: 30px; padding: 15px 0; border-top: 1px solid #30363d; color: #8b949e; font-size: 12px; }}
  .risk-high {{ color: #f85149; }}
  .risk-warn {{ color: #d2991d; }}
  .direction {{ display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }}
  .direction-long {{ background: #1b3d1b; color: #3fb950; }}
  .direction-short {{ background: #3d1b1b; color: #f85149; }}
</style>
</head>
<body>
<div class="container">
<h1>FVG KILLER — {session_label} 研究报告</h1>
<p style="color:#8b949e;">生成时间: {now.strftime('%Y-%m-%d %H:%M:%S')} (北京时间) | 追踪币种: {total} | 有信号: {with_signals} | 有分析: {with_analysis}</p>

<div class="summary">
  <div class="card">
    <div class="value">{total}</div>
    <div class="label">追踪币种</div>
  </div>
  <div class="card green">
    <div class="value">{with_signals}</div>
    <div class="label">有 FVG 信号</div>
  </div>
  <div class="card">
    <div class="value">{with_analysis}</div>
    <div class="label">多通道分析</div>
  </div>
  <div class="card green">
    <div class="value">{regime_counts.get('FUSED', 0)}</div>
    <div class="label">融合体制</div>
  </div>
  <div class="card red">
    <div class="value">{regime_counts.get('DIVERGENT', 0)}</div>
    <div class="label">背离体制</div>
  </div>
  <div class="card">
    <div class="value">{regime_counts.get('NEUTRAL', 0)}</div>
    <div class="label">中性</div>
  </div>
  <div class="card">
    <div class="value">{regime_counts.get('TRANSITIONING', 0)}</div>
    <div class="label">过渡中</div>
  </div>
</div>

<h2>Top {len(analyzed)} 币种综合研判</h2>
<table>
<thead>
<tr>
  <th>#</th>
  <th>币种</th>
  <th>价格</th>
  <th>方向</th>
  <th>综合评分</th>
  <th>置信度</th>
  <th>一致性</th>
  <th>体制</th>
  <th>信号数</th>
  <th>资金费率</th>
  <th>价差</th>
</tr>
</thead>
<tbody>
"""

    for i, entry in enumerate(analyzed):
        a = entry.analysis
        if a is None:
            continue

        score_class = "score-positive" if a.final_score >= 0 else "score-negative"
        direction = "做多" if a.final_score > 0.05 else ("做空" if a.final_score < -0.05 else "中性")
        dir_class = "direction-long" if a.final_score > 0.05 else ("direction-short" if a.final_score < -0.05 else "")
        regime = entry.detected_regime or "NEUTRAL"
        funding = f"{entry.funding_rate*100:+.4f}%" if entry.funding_rate is not None else "N/A"
        # 修复 R-1: spread_pct=0.0 是合法状态（零价差），不应显示为 N/A
        spread = f"{entry.spread_pct:.3f}%" if entry.spread_pct is not None else "N/A"

        risk_class = ""
        if entry.funding_rate is not None and abs(entry.funding_rate) > 0.005:
            risk_class = "risk-warn"
        if entry.funding_rate is not None and abs(entry.funding_rate) > 0.01:
            risk_class = "risk-high"

        html += f"""<tr>
  <td>{i+1}</td>
  <td><strong>{entry.inst_id}</strong></td>
  <td>{entry.current_price:.4f}</td>
  <td><span class="direction {dir_class}">{direction}</span></td>
  <td class="{score_class}">{a.final_score:+.2f}</td>
  <td>{(a.final_confidence or 0):.0%}</td>
  <td>{(a.channel_agreement or 0):.0%}</td>
  <td><span class="regime regime-{regime}">{regime}</span></td>
  <td>{len(entry.signals) if entry.signals else 0}</td>
  <td class="{risk_class}">{funding}</td>
  <td>{spread}</td>
</tr>
"""

    # 通道详情（只展示 Top 10）
    html += """
</tbody>
</table>

<h2>Top 10 通道详情</h2>
"""

    for i, entry in enumerate(analyzed[:10]):
        a = entry.analysis
        if a is None:
            continue
        direction = "做多" if a.final_score > 0.05 else ("做空" if a.final_score < -0.05 else "中性")

        html += f"""
<h3>#{i+1} {entry.inst_id} — {direction} | 置信度 {a.final_confidence:.0%} | 评分 {a.final_score:+.2f}</h3>
<table>
<thead><tr><th>通道</th><th>评分</th><th>置信度</th><th>摘要</th></tr></thead>
<tbody>
"""

        channel_names = ["价格行为", "市场结构", "资金流向", "市场情绪", "宏观背景"]
        for ch in a.channels:
            ch_summary = ", ".join(ch.observations[:2]) if ch.observations else ""
            html += f"""<tr>
  <td>{ch.channel_name}</td>
  <td class="{'score-positive' if ch.net_score >= 0 else 'score-negative'}">{ch.net_score:+.2f}</td>
  <td>{ch.confidence:.0%}</td>
  <td style="font-size:12px;color:#8b949e;">{ch_summary[:120]}</td>
</tr>"""

        html += "</tbody></table>"

    html += f"""
<div class="footer">
  <p>FVG KILLER（公允价值缺口杀手）v3.3 | 报告自动生成于 {now.strftime('%Y-%m-%d %H:%M:%S')} (北京时间)</p>
  <p>本报告仅供参考，不构成投资建议。加密货币交易风险极高，请谨慎决策。</p>
</div>
</div>
</body>
</html>"""

    return html


# ---------------------------------------------------------------------------
# 报告保存
# ---------------------------------------------------------------------------

def save_report(html_content: str, session_label: str, config: dict) -> str:
    """保存报告到本地文件。

    Returns:
        文件路径
    """
    report_cfg = config.get("report", {})
    report_dir = report_cfg.get("report_dir", "reports")
    os.makedirs(report_dir, exist_ok=True)

    now = datetime.now(BEIJING_TZ)
    filename = f"report_{now.strftime('%Y%m%d_%H%M')}_{session_label}.html"
    filepath = os.path.join(report_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"报告已保存: {filepath}")
    return filepath


# ---------------------------------------------------------------------------
# 邮件发送
# ---------------------------------------------------------------------------

def send_email_report(
    html_content: str,
    session_label: str,
    config: dict,
    attachment_path: Optional[str] = None,
) -> bool:
    """通过 SMTP 发送研究报告邮件。

    Args:
        html_content: HTML 报告内容
        session_label: 时段标签
        config: 完整配置
        attachment_path: 可选附件路径

    Returns:
        是否发送成功
    """
    email_cfg = config.get("report", {}).get("email", {})
    if not email_cfg.get("enabled", False):
        logger.info("邮件发送未启用，跳过")
        return False

    sender = email_cfg.get("sender", "")
    password = email_cfg.get("password", "")
    recipients = email_cfg.get("recipients", [])

    if not sender or not password or not recipients:
        logger.error("邮件配置不完整（sender/password/recipients），跳过发送")
        return False

    now = datetime.now(BEIJING_TZ)
    subject = f"[OKX FVG] {session_label} 研究报告 — {now.strftime('%Y-%m-%d %H:%M')}"

    try:
        msg = MIMEMultipart("alternative")
        msg["From"] = sender
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = subject
        # L-20: 强制英文 locale 防止 RFC 2822 日期头出现中文
        try:
            old_locale = locale.setlocale(locale.LC_TIME, 'C')
        except locale.Error:
            old_locale = None
        date_header = datetime.now(timezone.utc).strftime('%a, %d %b %Y %H:%M:%S +0000')
        if old_locale is not None:
            try:
                locale.setlocale(locale.LC_TIME, old_locale)
            except locale.Error:
                pass
        msg["Date"] = date_header

        # HTML 正文
        msg.attach(MIMEText(html_content, "html", "utf-8"))

        # 附件
        if attachment_path and os.path.exists(attachment_path):
            with open(attachment_path, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header(
                    "Content-Disposition",
                    f'attachment; filename="{os.path.basename(attachment_path)}"',
                )
                msg.attach(part)

        # 发送（带重试：最多 3 次，指数退避 2s/4s/8s）
        smtp_host = email_cfg.get("smtp_host", "smtp.qq.com")
        smtp_port = email_cfg.get("smtp_port", 465)
        max_retries = 3
        last_error = None

        for attempt in range(1, max_retries + 1):
            server = None
            try:
                if email_cfg.get("smtp_ssl", True):
                    server = smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=30)
                else:
                    server = smtplib.SMTP(smtp_host, smtp_port, timeout=30)
                    server.starttls()

                server.login(sender, password)
                server.sendmail(sender, recipients, msg.as_string())

                logger.info(f"邮件发送成功 → {', '.join(recipients)}")
                return True
            except (smtplib.SMTPConnectError, smtplib.SMTPException,
                    ConnectionError, TimeoutError, OSError) as e:
                last_error = e
                if attempt < max_retries:
                    delay = 2 ** attempt
                    logger.warning(
                        "SMTP 发送失败 (第 %d/%d 次): %s，%ds 后重试...",
                        attempt, max_retries, e, delay,
                    )
                    time.sleep(delay)
            finally:
                if server is not None:
                    try:
                        server.quit()
                    except Exception:
                        pass

        logger.error(f"邮件发送失败: 重试 {max_retries} 次后仍失败，最后错误: {last_error}")
        return False
    except Exception as e:
        logger.error(f"邮件构建/发送异常: {e}")
        return False


# ---------------------------------------------------------------------------
# 一站式入口
# ---------------------------------------------------------------------------

def generate_and_send_report(
    cache, config: dict, reporter: Optional[SessionReporter] = None
) -> bool:
    """生成报告、保存到本地、发送邮件（一站式）。

    Args:
        cache: CoinResearchCache 实例
        config: 完整配置
        reporter: 可选，外部 SessionReporter 实例（复用防重复触发状态）

    Returns:
        是否成功
    """
    # 修复: 优先使用外部传入的 reporter（复用 _triggered_minutes 防重复），
    # 仅在未传入时新建（兼容独立调用场景，如测试/手动触发）
    if reporter is None:
        reporter = SessionReporter(config)
        if not reporter.should_generate_report():
            return False

    session_label = reporter.get_session_label()
    logger.info(f"⏰ 触发 {session_label} 研究报告生成...")

    # 获取所有缓存条目
    entries = cache.get_all_entries()

    if not entries:
        logger.warning("缓存为空，跳过一次报告")
        return False

    # 生成 HTML
    html = generate_html_report(entries, session_label, config)

    # 保存到本地
    filepath = save_report(html, session_label, config)

    # 发送邮件
    email_sent = send_email_report(html, session_label, config, filepath)

    logger.info(
        f"报告生成完成: {session_label} | "
        f"条目: {len(entries)} | "
        f"保存: {filepath} | "
        f"邮件: {'已发送' if email_sent else '未发送'}"
    )

    return True