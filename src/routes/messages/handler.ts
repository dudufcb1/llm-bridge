// Anthropic /v1/messages endpoint -> translates to OpenAI and calls real OpenAI API

import { Hono } from "hono"
import consola from "consola"

import type { AnthropicMessagesPayload, AnthropicStreamState } from "~/lib/anthropic-types"
import type { ChatCompletionChunk } from "~/lib/openai-types"
import { callOpenAI } from "~/lib/api-client"
import { getConfig } from "~/lib/config"
import { translateToOpenAI, translateToAnthropic } from "./non-stream-translation"
import { translateChunkToAnthropicEvents, translateErrorToAnthropicErrorEvent } from "./stream-translation"

const app = new Hono()

app.post("/", async (c) => {
  const config = getConfig()
  const anthropicPayload = await c.req.json<AnthropicMessagesPayload>()

  consola.info(`[messages] model=${anthropicPayload.model} stream=${anthropicPayload.stream ?? false}`)

  let result: Awaited<ReturnType<typeof callOpenAI>>
  try {
    const openaiPayload = translateToOpenAI(anthropicPayload)
    result = await callOpenAI(openaiPayload, config.targetBaseUrl, config.targetApiKey)
  } catch (error) {
    const msg = error instanceof Error ? error.message : String(error)
    consola.error(`[messages] upstream error: ${msg}`)
    return c.json({ type: "error", error: { type: "api_error", message: msg } }, 500)
  }

  // Non-streaming
  if (!anthropicPayload.stream) {
    const anthropicResponse = translateToAnthropic(result as any)
    return c.json(anthropicResponse)
  }

  // Streaming
  const streamState: AnthropicStreamState = {
    messageStartSent: false,
    contentBlockIndex: 0,
    contentBlockOpen: false,
    thinkingBlockOpen: false,
    toolCalls: {},
  }

  const stream = new ReadableStream({
    async start(controller) {
      const encoder = new TextEncoder()

      const sendEvent = (event: { type: string; [key: string]: unknown }) => {
        controller.enqueue(encoder.encode(`event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`))
      }

      try {
        const eventStream = result as AsyncIterable<{ data: string }>
        for await (const event of eventStream) {
          if (!event.data || event.data === "[DONE]") continue

          let chunk: ChatCompletionChunk
          try {
            chunk = JSON.parse(event.data) as ChatCompletionChunk
          } catch {
            continue
          }

          const anthropicEvents = translateChunkToAnthropicEvents(chunk, streamState)
          for (const ae of anthropicEvents) {
            sendEvent(ae)
          }
        }
      } catch (error) {
        consola.error("Stream error:", error)
        sendEvent(translateErrorToAnthropicErrorEvent())
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
