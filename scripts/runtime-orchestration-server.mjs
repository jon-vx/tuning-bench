#!/usr/bin/env node

import { createHash } from "node:crypto";
import { createReadStream, existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const port = Number(process.argv[2] || 8000);
const mime = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".md": "text/markdown; charset=utf-8",
  ".svg": "image/svg+xml",
};

function send(response, status, body, type = "text/plain; charset=utf-8") {
  response.writeHead(status, { "content-type": type, "cache-control": "no-store" });
  response.end(body);
}

function safeCampaignId(value) {
  const id = String(value || "").toLowerCase().replace(/[^a-z0-9-]+/g, "-").replace(/^-|-$/g, "");
  if (!id || id.length > 100) throw new Error("invalid campaignId");
  return id;
}

function savePayload(payload) {
  if (payload?.schemaVersion !== "webllm-runtime-orchestration-v1") {
    throw new Error("unexpected schemaVersion");
  }
  if (!Array.isArray(payload.rows) || !Array.isArray(payload.summaries)) {
    throw new Error("rows and summaries are required");
  }
  const campaignId = safeCampaignId(payload.campaignId);
  const directory = path.join(root, "..", "data", "runtime-orchestration", campaignId);
  const file = path.join(directory, "raw-results.json");
  if (existsSync(file)) throw new Error(`campaign already exists: ${campaignId}`);
  const text = JSON.stringify(payload, null, 2) + "\n";
  const sha256 = createHash("sha256").update(text).digest("hex");
  mkdirSync(directory, { recursive: true });
  writeFileSync(file, text);
  writeFileSync(path.join(directory, "raw-results.sha256"), `${sha256}  raw-results.json\n`);
  return {
    campaignId,
    rows: payload.rows.length,
    failures: payload.rows.filter((row) => !row.success).length,
    sha256,
    path: path.relative(root, file),
  };
}

const server = createServer((request, response) => {
  if (request.method === "POST" && request.url === "/api/runtime-orchestration-results") {
    let body = "";
    request.setEncoding("utf8");
    request.on("data", (chunk) => {
      body += chunk;
      if (body.length > 20_000_000) request.destroy();
    });
    request.on("end", () => {
      try {
        send(response, 201, JSON.stringify(savePayload(JSON.parse(body))), "application/json; charset=utf-8");
      } catch (error) {
        send(response, 400, JSON.stringify({ error: error.message }), "application/json; charset=utf-8");
      }
    });
    return;
  }
  if (request.method !== "GET" && request.method !== "HEAD") {
    send(response, 405, "method not allowed");
    return;
  }
  const urlPath = decodeURIComponent(new URL(request.url, `http://${request.headers.host}`).pathname);
  const relative = urlPath === "/" ? "index.html" : urlPath.replace(/^\/+/, "");
  const file = path.resolve(root, relative);
  if (!file.startsWith(root + path.sep) || !existsSync(file) || !statSync(file).isFile()) {
    send(response, 404, "not found");
    return;
  }
  response.writeHead(200, {
    "content-type": mime[path.extname(file)] || "application/octet-stream",
    "content-length": statSync(file).size,
    "cache-control": "no-store",
  });
  if (request.method === "HEAD") response.end();
  else createReadStream(file).pipe(response);
});

server.listen(port, "0.0.0.0", () => {
  process.stdout.write(`Runtime experiment server: http://127.0.0.1:${port}\n`);
});
