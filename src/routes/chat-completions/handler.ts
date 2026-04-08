// OpenAI /v1/chat/completions endpoint -> translates to Anthropic and calls real Anthropic API

import { Hono } from "hono"
import consola from "consola"

import type { ChatCompletionsPayload } from "~/lib/openai-types"
import { callAnthropic } from "~/lib/api-client"
import { getConfig } from "~/lib/config"
import {
  translateToAnthropic,
  translateToOpenAI,
} from "./openai-to-anthropic"
import {
  translateAnthropicEventToOpenAIChunks,
  createOpenAIStreamState,
} from "./stream-translation"

const app = new Hono()

app.post("/", async (c) => {
  const config = getConfig()
  const openaiPayload = await c.req.json<ChatCompletionsPayload>()

  consola.info(`[chat-completions] model=${openaiPayload.model} stream=${openaiPayload.stream ?? false}`)

  let result: Awaited<ReturnType<typeof callAnthropic>>
  try {
    const anthropicPayload = translateToAnthropic(openaiPayload)
    result = await callAnthropic(anthropicPayload, config.targetBaseUrl, config.targetApiKey)
  } catch (error) {
    const msg = error instanceof Error ? error.message : String(error)
    consola.error(`[chat-completions] upstream error: ${msg}`)
    return c.json({ error: { message: msg, type: "api_error" } }, 500)
  }

  // Non-streaming
  if (!openaiPayload.stream) {
    const openaiResponse = translateToOpenAI(result as any)
    return c.json(openaiResponse)
  }

  // Streaming
  const streamState = createOpenAIStreamState()

  const stream = new ReadableStream({
    async start(controller) {
      const encoder = new TextEncoder()

      const sendChunk = (data: string) => {
        controller.enqueue(encoder.encode(`data: ${data}\n\n`))
      }

      try {
        const eventStream = result as AsyncIterable<{ event?: string; data: string }>
        for await (const event of eventStream) {
          if (!event.data || event.data === "[DONE]") continue

          const openaiChunks = translateAnthropicEventToOpenAIChunks(
            { event: event.event || "", data: event.data },
            streamState,
          )
          for (const chunk of openaiChunks) {
            sendChunk(JSON.stringify(chunk))
          }
        }

        sendChunk("[DONE]")
      } catch (error) {
        consola.error("Stream error:", error)
        sendChunk(JSON.stringify({
          error: { message: "Stream translation error", type: "server_error" },
        }))
      } finally {
        controller.close()
      }
    },
  })

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "Connection": "keep-alive",
    },
  })
})

export default app
