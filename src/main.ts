import { defineCommand, runMain } from "citty"
import consola from "consola"

const main = defineCommand({
  meta: {
    name: "llm-bridge",
    description: "Lightweight bidirectional API translator between OpenAI and Anthropic formats",
  },
  args: {
    port: {
      type: "string",
      description: "Port to listen on",
      default: "4141",
    },
    "target-base-url": {
      type: "string",
      description: "Target API base URL (auto-detected from direction if not set)",
    },
    "target-api-key": {
      type: "string",
      description: "API key for the target provider",
    },
    "api-key": {
      type: "string",
      description: "API key to protect this proxy (optional)",
    },
    verbose: {
      type: "boolean",
      description: "Enable verbose logging",
      default: false,
    },
  },
  async run({ args }) {
    // Set env vars from CLI args (config.ts reads from env)
    if (args.port) process.env.PORT = args.port
    if (args["target-base-url"]) process.env.TARGET_BASE_URL = args["target-base-url"]
    if (args["target-api-key"]) process.env.TARGET_API_KEY = args["target-api-key"]
    if (args["api-key"]) process.env.API_KEY = args["api-key"]
    if (args.verbose) process.env.VERBOSE = "true"

    const { createServer } = await import("~/server")
    const { getConfig } = await import("~/lib/config")

    const config = getConfig()
    const app = createServer()

    consola.info("llm-bridge starting...")
    consola.info(`  Port: ${config.port}`)
    consola.info(`  Target: ${config.targetBaseUrl}`)
    consola.info(`  Target API key: ${config.targetApiKey ? "***" + config.targetApiKey.slice(-4) : "(none)"}`)
    consola.info("")
    consola.info("Endpoints:")
    consola.info("  POST /v1/messages          (Anthropic format -> forwards to OpenAI)")
    consola.info("  POST /v1/chat/completions  (OpenAI format -> forwards to Anthropic)")
    consola.info("")

    Bun.serve({
      port: config.port,
      fetch: app.fetch,
    })

    consola.success(`llm-bridge listening on http://localhost:${config.port}`)
  },
})

runMain(main)
