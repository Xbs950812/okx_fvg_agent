/**
 * FVG KILLER — Royalty 匿名统计端点 (Cloudflare Worker + KV, 无需 Telegram)
 *
 * 三个功能合一:
 *   POST /report        接收部署实例的匿名心跳/事件, 存入 KV
 *   GET  /admin?key=X   作者看板: 全部部署实例的表格 (key = ADMIN_KEY 环境变量)
 *   GET  /              健康检查
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
 * 不收集: API密钥/钱包/持仓/交易细节/任何个人信息
 * (服务端仅能看到必然暴露的连接IP, 且不存储)
 *
 * ── 部署步骤 (作者一次性操作, ~15分钟) ─────────────────────
 * 1. Cloudflare Dashboard → Workers 和 Pages → KV → 创建命名空间 `ROYALTY`
 * 2. Workers 和 Pages → 创建 Worker (名字如 fvg-report) → Deploy
 * 3. 编辑代码: 全选删除, 粘贴本文件 → Deploy
 * 4. Worker Settings:
 *      a) Variables → 添加 ADMIN_KEY = 自选随机串 (Secret 类型) — 看板访问凭证
 *      b) Bindings → KV 绑定: 变量名 ROYALTY_KV, 命名空间选 ROYALTY
 * 5. 激活: config.json → royalty.report_url = https://<worker>.workers.dev/report
 *    重启 agent 后浏览器打开 https://<worker>.workers.dev/admin?key=<ADMIN_KEY>
 *
 * 验证: curl -X POST https://<worker>.workers.dev/report \
 *   -H "content-type: application/json" \
 *   -d '{"install_id":"test123","event":"heartbeat","version":"3.3.0","paper_mode":true,"pool_usdt":0,"cumulative_royalty_usdt":0,"withdrawals_count":0,"ts":1}'
 * → 然后打开 /admin?key=... 应看到 test123 这一行
 *
 * 免费额度: KV 免费 10万读/天 + 1000写/天, 远超本项目场景需求 (每实例每天 1 条心跳)
 */

const ALLOWED_FIELDS = [
  "install_id", "version", "ts", "event", "paper_mode",
  "pool_usdt", "cumulative_royalty_usdt", "withdrawals_count",
];
const VALID_EVENTS = new Set(["heartbeat", "withdrawal", "perm_denied"]);
// KV 每个实例的记录: { latest: {...payload}, history: [最近N条事件] }
const HISTORY_LIMIT = 50;          // 每实例保留最近 50 条事件
const PRUNE_DAYS = 90;             // 90 天无心跳的实例在看板标记为失联

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (request.method === "GET" && url.pathname === "/admin") {
      return handleAdmin(url, env);
    }
    if (request.method === "GET") {
      return new Response("FVG KILLER royalty report endpoint. POST /report", {
        status: 200,
      });
    }
    if (request.method !== "POST" || url.pathname !== "/report") {
      return new Response("not found", { status: 404 });
    }
    return handleReport(request, env);
  },
};

// ---------------------------------------------------------------- POST /report

async function handleReport(request, env) {
  let body;
  try {
    body = await request.json();
  } catch {
    return new Response("bad json", { status: 400 });
  }

  // 字段白名单过滤
  const payload = {};
  for (const k of ALLOWED_FIELDS) {
    if (k in body) payload[k] = body[k];
  }
  if (!payload.install_id || !payload.event
      || !VALID_EVENTS.has(payload.event)) {
    return new Response("bad payload", { status: 400 });
  }

  // 读旧记录 → 更新 latest + 追加 history (每实例一个 KV key)
  const kvKey = `inst:${payload.install_id}`;
  let rec;
  try {
    rec = (await env.ROYALTY_KV.get(kvKey, "json")) || { latest: null, history: [] };
  } catch {
    rec = { latest: null, history: [] };
  }
  rec.latest = payload;
  rec.history.unshift({ event: payload.event, ts: payload.ts || Date.now() / 1000 });
  if (rec.history.length > HISTORY_LIMIT) {
    rec.history = rec.history.slice(0, HISTORY_LIMIT);
  }

  try {
    await env.ROYALTY_KV.put(kvKey, JSON.stringify(rec));
  } catch (e) {
    return new Response("kv-fail", { status: 200 }); // 仍返回200, 客户端无需重试
  }
  return new Response("ok", { status: 200 });
}

// ---------------------------------------------------------------- GET /admin

async function handleAdmin(url, env) {
  // 鉴权: /admin?key=<ADMIN_KEY>  (鉴权失败不提示原因, 统一 404)
  const key = url.searchParams.get("key") || "";
  if (!env.ADMIN_KEY || key !== env.ADMIN_KEY) {
    return new Response("not found", { status: 404 });
  }

  // 列出全部实例记录 (免费版 list 每页 1000 key, 足够)
  const rows = [];
  let cursor;
  try {
    do {
      const page = await env.ROYALTY_KV.list({ cursor });
      const keys = page.keys.filter((k) => k.name.startsWith("inst:"));
      const values = await Promise.all(
        keys.map((k) => env.ROYALTY_KV.get(k.name, "json")));
      for (const v of values) {
        if (v && v.latest) rows.push(v);
      }
      cursor = page.list_complete ? undefined : page.cursor;
    } while (cursor);
  } catch (e) {
    return new Response("kv list fail: " + e, { status: 500 });
  }

  // 聚合统计
  const total = rows.length;
  const live = rows.filter((r) => !r.latest.paper_mode).length;
  const paper = total - live;
  const totalPool = rows.reduce((s, r) => s + (+r.latest.pool_usdt || 0), 0);
  const totalRoyalty = rows.reduce(
    (s, r) => s + (+r.latest.cumulative_royalty_usdt || 0), 0);
  const totalWd = rows.reduce(
    (s, r) => s + (+r.latest.withdrawals_count || 0), 0);
  const permDenied = rows.filter((r) =>
    r.history.some((h) => h.event === "perm_denied")).length;

  // 按最近上报时间倒序
  rows.sort((a, b) => (b.latest.ts || 0) - (a.latest.ts || 0));

  const cutoff = Date.now() / 1000 - PRUNE_DAYS * 86400;
  const fmtTime = (ts) => {
    const d = new Date((ts || 0) * 1000);
    const p = (n) => String(n).padStart(2, "0");
    return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ` +
           `${p(d.getHours())}:${p(d.getMinutes())}`;
  };
  const esc = (s) => String(s).replace(/[&<>"]/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  const trs = rows.map((r) => {
    const l = r.latest;
    const stale = (l.ts || 0) < cutoff;
    const lastEvents = r.history.slice(0, 5)
      .map((h) => `${h.event}@${fmtTime(h.ts)}`).join("<br>");
    return `<tr class="${stale ? "stale" : ""}">
      <td class="mono">${esc(l.install_id)}</td>
      <td>${l.paper_mode ? "纸面" : "实盘"}</td>
      <td class="num">${(+l.pool_usdt || 0).toFixed(2)}</td>
      <td class="num">${(+l.cumulative_royalty_usdt || 0).toFixed(2)}</td>
      <td class="num">${+l.withdrawals_count || 0}</td>
      <td class="mono">${esc(l.version || "?")}</td>
      <td class="events">${lastEvents}</td>
      <td>${stale ? '<span class="stale-tag">失联</span> ' : ""}${fmtTime(l.ts)}</td>
    </tr>`;
  }).join("\n");

  const html = `<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex,nofollow">
<title>FVG KILLER — Royalty 看板</title>
<style>
  body { font-family: -apple-system, "Segoe UI", sans-serif; background:#0d1117;
         color:#e6edf3; margin:0; padding:24px; }
  h1 { font-size:20px; margin:0 0 4px; }
  .sub { color:#8b949e; font-size:13px; margin-bottom:20px; }
  .cards { display:flex; gap:12px; flex-wrap:wrap; margin-bottom:20px; }
  .card { background:#161b22; border:1px solid #30363d; border-radius:8px;
          padding:12px 18px; min-width:130px; }
  .card .v { font-size:22px; font-weight:700; color:#58a6ff; }
  .card .l { font-size:12px; color:#8b949e; margin-top:2px; }
  table { border-collapse:collapse; width:100%; font-size:13px; }
  th, td { padding:8px 10px; border-bottom:1px solid #21262d; text-align:left; }
  th { color:#8b949e; font-weight:600; background:#161b22; position:sticky; top:0; }
  tr:hover td { background:#161b2288; }
  .mono { font-family:ui-monospace,monospace; }
  .num { text-align:right; font-family:ui-monospace,monospace; }
  .events { font-size:11px; color:#8b949e; line-height:1.6; }
  .stale { opacity:0.45; }
  .stale-tag { background:#da3633; color:#fff; border-radius:4px;
               font-size:10px; padding:1px 5px; }
  .empty { color:#8b949e; padding:40px; text-align:center; }
  .refresh { color:#58a6ff; font-size:12px; }
</style>
<meta http-equiv="refresh" content="300">
</head>
<body>
<h1>FVG KILLER — Royalty 看板</h1>
<div class="sub">匿名部署统计 · 每 5 分钟自动刷新 · 数据保留 ${HISTORY_LIMIT} 条事件/实例</div>
<div class="cards">
  <div class="card"><div class="v">${total}</div><div class="l">总部署</div></div>
  <div class="card"><div class="v">${live}</div><div class="l">实盘</div></div>
  <div class="card"><div class="v">${paper}</div><div class="l">纸面</div></div>
  <div class="card"><div class="v">${totalPool.toFixed(2)}</div><div class="l">池余额合计 (USDT)</div></div>
  <div class="card"><div class="v">${totalRoyalty.toFixed(2)}</div><div class="l">累计分成合计 (USDT)</div></div>
  <div class="card"><div class="v">${totalWd}</div><div class="l">累计提现笔数</div></div>
  <div class="card"><div class="v">${permDenied}</div><div class="l">权限被拒实例</div></div>
</div>
${total === 0 ? '<div class="empty">还没有任何上报。<br>激活一个实例后等 24h 内首条心跳, 或 curl 手动 POST /report 测试。</div>' : `
<table>
<thead><tr>
  <th>install_id</th><th>模式</th><th>池余额</th><th>累计分成</th>
  <th>提现</th><th>版本</th><th>最近事件</th><th>最近上报</th>
</tr></thead>
<tbody>
${trs}
</tbody>
</table>`}
<p class="refresh">统计口径: 累计分成 ×10 ≈ 该实例历史盈利总额 · 失联 = ${PRUNE_DAYS} 天无心跳</p>
</body>
</html>`;

  return new Response(html, {
    status: 200,
    headers: { "content-type": "text/html; charset=utf-8" },
  });
}
