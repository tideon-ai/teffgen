# teffgen-client

Official TypeScript / JavaScript client for the [tideon.ai](https://tideon.ai) API server.

Works in Node 18+, Deno, Bun, and modern browsers. Uses the built-in
`fetch` and `ReadableStream` — no extra runtime dependencies.

## Install

```bash
npm install teffgen-client
```

## Usage

```ts
import { TeffgenClient } from "teffgen-client";

const client = new TeffgenClient({
  baseUrl: "http://localhost:8000",
  apiKey: process.env.TEFFGEN_API_KEY,
});

// Simple chat
const res = await client.chat("What is 2+2?", { tools: ["calculator"] });
console.log(res.content);

// Streaming chat
for await (const chunk of client.chatStream("Tell me a story")) {
  process.stdout.write(chunk);
}

// Embeddings
const vecs = await client.embed(["Hello", "World"]);
console.log(vecs.length, vecs[0].length);

// Health
const health = await client.health();
console.log(health.ok);
```

## Error handling

All errors derive from `TeffgenClientError`:

- `TeffgenConnectionError` — network failure
- `TeffgenTimeoutError` — request timed out
- `TeffgenAPIError` — non-2xx response
  - `TeffgenAuthError` — 401 / 403
  - `TeffgenRateLimitError` — 429
  - `TeffgenServerError` — 5xx

Connection, timeout, 429, and 5xx errors are retried automatically with
exponential backoff (configurable via `maxRetries` / `backoffBaseMs`).

## License

Apache-2.0
