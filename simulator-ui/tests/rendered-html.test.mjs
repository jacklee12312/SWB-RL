import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders the SWB simulator shell", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>SWB 对局模拟器<\/title>/i);
  assert.match(html, /正在载入对局引擎/);
  assert.match(html, /读取卡组、规则、历史记录和 PPO 策略/);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape/);
});

test("ships product UI and removes starter-only assets", async () => {
  const [page, layout, css, packageJson] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
  ]);

  assert.match(page, /\/api\/new-match/);
  assert.match(page, /\/api\/action/);
  assert.match(page, /\/api\/history/);
  assert.match(page, /原始对局日志/);
  assert.match(page, /AI LAST TURN/);
  assert.match(page, /主战者区域/);
  assert.match(page, /RESOLUTION/);
  assert.match(page, /对局记录/);
  assert.match(page, /AI 决策分布/);
  assert.match(page, /当时 AI 手牌/);
  assert.match(page, /historyRevealsPrivateInformation/);
  assert.match(page, /candidate\.probability/);
  assert.match(page, /原始结算日志/);
  assert.match(page, /觉醒/);
  assert.match(page, /墓影/);
  assert.match(page, /ai\.hand_count/);
  assert.match(page, /human\.hand\?\.map/);
  assert.match(page, /card\.union_bursts/);
  assert.match(page, /奥义进度/);
  assert.match(page, /解放奥义/);
  assert.match(layout, /title:\s*"SWB 对局模拟器"/);
  assert.match(css, /\.game-layout/);
  assert.match(css, /\.card-tile/);
  assert.match(css, /\.card-burst-progress/);
  assert.match(css, /\.card-burst-badge\.ready/);
  assert.match(css, /\.leader-zone-slots/);
  assert.match(css, /\.battle-broadcast/);
  assert.match(css, /\.history-drawer/);
  assert.match(css, /\.history-policy-actions/);
  assert.match(css, /\.history-raw-logs/);
  assert.match(css, /@media \(max-width: 720px\)/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);

  await assert.rejects(
    access(new URL("../app/_sites-preview/SkeletonPreview.tsx", import.meta.url)),
  );
});
