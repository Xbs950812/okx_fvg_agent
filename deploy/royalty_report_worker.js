/**
 * FVG KILLER — Royalty 匿名统计端点 (Cloudflare Worker, 免费版即可)
 *
 * 用途: 接收各部署实例的匿名心跳/事件, 格式化后转发到作者 Telegram。
 *       本文件公开在仓库中 = 任何人可审计收集了什么、没收集什么。
 *
 * 接收字段 (仅此 8 个, 超出的一律丢弃):
 *   install_id                匿名安装ID (客户端首次随机生成, 12位hex)
 *   version                   代码版本
 *   ts                        客户端时间戳
 *   event                     heartbeat | withdrawal | perm_denied
 *   paper_mode                是否纸面/模拟模式
 *   pool_usdt                 分成池当前余额
 *   cumulative_royalty_usdt   历史累计分成 (= 盈利总额的10%)
 *   withdrawals_count         已完成提现笔数
 *
 * 不收集: API密钥/钱包/持仓/交易细节/任何个人信息 (服务端仅能看到必然暴露的连接IP)
 *
 * ── 部署步骤 (作者一次性操作, ~10分钟) ─────────────────────
 * 1. Telegram 找 @BotFather → /newbot → 得到 BOT_TOKEN
 * 2. 给你的 bot 发一条消息, 浏览器打开
 *    https://api.telegram.org/bot<BOT_TOKEN>/getUpdates → 取 result[0].message.chat.id
 * 3. Cloudflare Dashboard → Workers & Pages → Create Worker → 粘贴本文件 → Deploy
 * 4. Worker Settings → Variables:
 *      TG_TOKEN   = <BOT_TOKEN>        (Secret 类型)
 *      TG_CHAT_ID = <chat_id>
 * 5. 得到 Worker URL (形如 https://xxx.workers.dev), 在路径后加 /report
 *    填入 config.json → royalty.report_url
 *
 * 验证: curl -X POST https://xxx.workers.dev/report \
 *   -H "content-type: application/json" \
 *   -d '{"install_id":"test123","event":"heartbeat","version":"3.3.0"}'
 * → Telegram 应收到消息
 */

const ALLOWED_FIELDS = [
  "install_id", "version", "ts", "event", "paper_mode",
  "pool_usdt", "cumulative_royalty_usdt", "withdrawals_count",
];
const VALID_EVENTS = new Set(["heartbeat", "withdrawal", "perm_denied"]);

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // 健康检查
    if (request.method === "GET") {
      return new Response("FVG KILLER royalty report endpoint. POST /report", {
        status: 200,
      });
    }
    if (request.method !== "POST" || url.pathname !== "/report") {
      return new Response("not found", { status: 404 });
    }

    // 解析 + 字段白名单过滤
    let body;
    try {
      body = await request.json();
    } catch {
      return new Response("bad json", { status: 400 });
    }
    const payload = {};
    for (const k of ALLOWED_FIELDS) {
      if (k in body) payload[k] = body[k];
    }
    if (!payload.install_id || !payload.event
        || !VALID_EVENTS.has(payload.event)) {
      return new Response("bad payload", { status: 400 });
    }

    // 格式化为 Telegram 消息 (中文, 便于作者直接阅读)
    const paper = payload.paper_mode ? "纸面" : "实盘";
    const msg =
      `[FVG-KILLER] ${payload.event}\n` +
      `id: ${payload.install_id}  v${payload.version || "?"}  (${paper})\n` +
      `池: ${payload.pool_usdt ?? "-"} USDT  ` +
      `累计分成: ${payload.cumulative_royalty_usdt ?? "-"} USDT  ` +
      `已提现: ${payload.withdrawals_count ?? 0} 笔`;

    try {
      await fetch(`https://api.telegram.org/bot${env.TG_TOKEN}/sendMessage`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ chat_id: env.TG_CHAT_ID, text: msg }),
      });
    } catch (e) {
      return new Response("tg-fail", { status: 200 }); // 仍返回200, 客户端无需重试
    }
    return new Response("ok", { status: 200 });
  },
};
