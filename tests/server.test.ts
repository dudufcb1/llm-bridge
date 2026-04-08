import { describe, expect, test, beforeAll, afterAll } from "bun:test"
import { createServer } from "~/server"

describe("server", () => {
  test("GET /health returns ok", async () => {
    const app = createServer()
    const res = await app.request("/health")

    expect(res.status).toBe(200)
    const body = await res.json()
    expect(body.status).toBe("ok")
  })

  test("GET /v1/models returns list", async () => {
    const app = createServer()
    const res = await app.request("/v1/models")

    expect(res.status).toBe(200)
    const body = await res.json()
    expect(body.object).toBe("list")
    expect(body.data).toBeArray()
  })

  test("rejects unauthorized request when API_KEY is set", async () => {
    process.env.API_KEY = "test-secret-key"
    const app = createServer()

    const res = await app.request("/v1/models", {
      headers: { "Authorization": "Bearer wrong-key" },
    })

    expect(res.status).toBe(401)

    // Clean up
    delete process.env.API_KEY
  })

  test("allows authorized request with correct API_KEY", async () => {
    process.env.API_KEY = "test-secret-key"
    const app = createServer()

    const res = await app.request("/v1/models", {
      headers: { "Authorization": "Bearer test-secret-key" },
    })

    expect(res.status).toBe(200)

    delete process.env.API_KEY
  })

  test("allows request with x-api-key header", async () => {
    process.env.API_KEY = "test-secret-key"
    const app = createServer()

    const res = await app.request("/v1/models", {
      headers: { "x-api-key": "test-secret-key" },
    })

    expect(res.status).toBe(200)

    delete process.env.API_KEY
  })

  test("allows request when no API_KEY configured", async () => {
    delete process.env.API_KEY
    const app = createServer()

    const res = await app.request("/v1/models")
    expect(res.status).toBe(200)
  })

  test("POST /v1/messages returns error without valid target", async () => {
    process.env.TARGET_BASE_URL = "http://localhost:19999"
    process.env.TARGET_API_KEY = "fake-key"
    const app = createServer()

    const res = await app.request("/v1/messages", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "gpt-4o",
        messages: [{ role: "user", content: "Hi" }],
        max_tokens: 100,
      }),
    })

    // Should fail because target is unreachable
    expect(res.status).toBe(500)

    delete process.env.TARGET_BASE_URL
    delete process.env.TARGET_API_KEY
  })

  test("POST /v1/chat/completions returns error without valid target", async () => {
    process.env.TARGET_BASE_URL = "http://localhost:19999"
    process.env.TARGET_API_KEY = "fake-key"
    const app = createServer()

    const res = await app.request("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "claude-sonnet-4-20250514",
        messages: [{ role: "user", content: "Hi" }],
        max_tokens: 100,
      }),
    })

    expect(res.status).toBe(500)

    delete process.env.TARGET_BASE_URL
    delete process.env.TARGET_API_KEY
  })
})
