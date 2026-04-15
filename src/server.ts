import { Hono } from "hono"
import { cors } from "hono/cors"
import consola from "consola"

import { getConfig } from "~/lib/config"
import messagesHandler from "~/routes/messages/handler"
import chatCompletionsHandler from "~/routes/chat-completions/handler"
import responsesHandler from "~/routes/responses/handler"

export function createServer() {
  const config = getConfig()
  const app = new Hono()

  app.use("*", cors())

  // Auth middleware
  app.use("*", async (c, next) => {
    if (config.apiKey) {
      const auth = c.req.header("authorization") || c.req.header("x-api-key")
      const token = auth?.replace(/^Bearer\s+/i, "")
      if (token !== config.apiKey && token !== config.targetApiKey) {
        return c.json({ error: { message: "Unauthorized", type: "auth_error" } }, 401)
      }
    }
    await next()
  })

  // Health check
  app.get("/health", (c) => c.json({ status: "ok", direction: config.direction }))

  // All directions available
  app.route("/v1/messages", messagesHandler)
  app.route("/v1/chat/completions", chatCompletionsHandler)
  app.route("/v1/responses", responsesHandler)

  // Models endpoint - return configured model
  app.get("/v1/models", (c) => {
    return c.json({
      object: "list",
      data: [
        {
          id: "moonshotai/kimi-k2.5",
          object: "model",
          created: 1700000000,
          owned_by: "moonshotai",
        },
      ],
    })
  })

  return app
}
