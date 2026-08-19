/**
 * FVG KILLER — Royalty 统计端点 (Cloudflare Worker → Server酱微信推送)
 *
 * 三个路由:
 *   POST /report   接收部署实例匿名心跳/事件 (8 字段白名单)
 *   GET  /         健康检查
 *   (历史版本曾含 /digest 与 /debug, 见 git log; 正式版仅事件推送)
 *
 * 推送策略 (心跳静默省额度, 重要事件即时推):
 *   heartbeat    → 不推送
 *   withdrawal   → 「💰 FVG分成提现」
 *   perm_denied  → 「🔑 FVG权限被拒」
 *
 * 接收字段 (仅此 8 个, 超出的一律丢弃):
 *   install_id / version / ts / event / paper_mode
 *   pool_usdt / cumulative_royalty_usdt / withdrawals_count
 *
 * 不收集: API密钥/钱包/持仓/交易细节/任何个人信息
 *
 * ── 部署步骤 (作者一次性操作, ~10分钟) ─────────────────────
 * 1. sct.ftqq.com 微信扫码登录 → SendKey 页复制 key (SCT 开头)
 * 2. Cloudflare Workers → Create Worker (名字如 fvg-report) → Deploy
 * 3. Edit code: 全选删除默认代码, 粘贴本文件 → Deploy
 * 4. Settings → Variables and secrets → Add variable:
 *      Type=Secret, Name=SCT_SENDKEY, Value=你的 SendKey → Save
 * 5. 验证: 浏览器开 https://<worker>.workers.dev 应显示 endpoint 文案;
 *    curl -X POST .../report -H "content-type: application/json" \
 *      -d '{"install_id":"test","event":"withdrawal","version":"3.3.0",
 *           "ts":1,"paper_mode":false,"pool_usdt":0,
 *           "cumulative_royalty_usdt":0,"withdrawals_count":0}'
 *    → 微信「方糖」服务号收到「💰 FVG分成提现」即全链路通
 * 6. config.json → royalty.report_url = https://<worker>.workers.dev/report
 *
 * 实测记录 (2026-08-19, 账号 Pro 订阅):
 *   - .send 接口 200 ≠ 送达: 返回体 data.pushid/readkey 可查真实状态,
 *     调试期务必透传响应体 (曾因 try/catch 吞错 + 未完整 await 排查 2h)
 *   - sctapi.ftqq.com 从国内直连与代理均可达 (200);
 *     Worker 海外出口访问 sctapi 正常 (debug 直发验证)
 *   - Server酱 后台「消息记录」只显示最近 24h, 排查推送未达先看这里
 */

const ALLOWED_FIELDS = [
  "install_id", "version", "ts", "event", "paper_mode",
  "pool_usdt", "cumulative_royalty_usdt", "withdrawals_count",
];
const VALID_EVENTS = new Set(["heartbeat", "withdrawal", "perm_denied"]);

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (request.method === "GET") {
      return new Response("FVG KILLER royalty report endpoint. POST /report", { status: 200 });
    }
    if (request.method !== "POST" || url.pathname !== "/report") {
      return new Response("not found", { status: 404 });
    }
    let body;
    try { body = await request.json(); } catch { return new Response("bad json", { status: 400 }); }
    const payload = {};
    for (const k of ALLOWED_FIELDS) if (k in body) payload[k] = body[k];
    if (!payload.install_id || !payload.event || !VALID_EVENTS.has(payload.event)) {
      return new Response("bad payload", { status: 400 });
    }
    // 心跳不推送(省额度), 仅事件推送
    if (payload.event === "heartbeat") return new Response("ok", { status: 200 });

    const paper = payload.paper_mode ? "纸面" : "实盘";
    const title = payload.event === "withdrawal" ? "💰 FVG分成提现" : "🔑 FVG权限被拒";
    const desp = `事件: ${payload.event}\n实例: ${payload.install_id} (${paper}) v${payload.version || "?"}\n池: ${payload.pool_usdt} USDT\n累计分成: ${payload.cumulative_royalty_usdt} USDT\n已提现: ${payload.withdrawals_count ?? 0} 笔`;
    try {
      const r = await fetch(`https://sctapi.ftqq.com/${env.SCT_SENDKEY}.send`, {
        method: "POST",
        headers: { "content-type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams({ title, desp }),
      });
      await r.text();
    } catch (e) {
      // 推送失败不阻塞客户端
    }
    return new Response("ok", { status: 200 });
  },
};

