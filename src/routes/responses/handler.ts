// Anthropic /v1/messages endpoint -> translates to OpenAI Responses API
// Used when the target model supports Responses API (reasoning models like o5.4+)

import { Hono } from "hono"
import consola from "consola"

import type { AnthropicMessagesPayload } from "~/lib/anthropic-types"
import type { ResponseStreamEvent, ResponsesResult } from "~/lib/responses-types"
import { callOpenAIResponses } from "~/lib/api-client"
import { getConfig } from "~/lib/config"
import { translateAnthropicToResponsesPayload, translateResponsesToAnthropic } from "./translation"
import {
  translateResponsesStreamEvent,
  createResponsesStreamState,
  buildErrorEvent,
} from "./stream-translation"

const app = new Hono()

app.post("/", async (c) => {
  const config = getConfig()
  const anthropicPayload = await c.req.json<AnthropicMessagesPayload>()

  consola.info(`[responses] model=${anthropicPayload.model} stream=${anthropicPayload.stream ?? false}`)

  let result: Awaited<ReturnType<typeof callOpenAIResponses>>
  try {
    const responsesPayload = translateAnthropicToResponsesPayload(anthropicPayload)
    result = await callOpenAIResponses(responsesPayload, config.targetBaseUrl, config.targetApiKey)
  } catch (error) {
    const msg = error instanceof Error ? error.message : String(error)
    consola.error(`[responses] upstream error: ${msg}`)
    return c.json({ type: "error", error: { type: "api_error", message: msg } }, 500)
  }

  // Non-streaming
  if (!anthropicPayload.stream) {
    const anthropicResponse = translateResponsesToAnthropic(result as ResponsesResult)
    return c.json(anthropicResponse)
  }

  // Streaming
  const streamState = createResponsesStreamState()

  const stream = new ReadableStream({
    async start(controller) {
      const encoder = new TextEncoder()

      const sendEvent = (event: { type: string; [key: string]: unknown }) => {
        controller.enqueue(encoder.encode(`event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`))
      }

      try {
        const eventStream = result as AsyncIterable<{ event?: string; data: string }>
        for await (const event of eventStream) {
          if (!event.data || event.data === "[DONE]") continue

          let parsed: ResponseStreamEvent
          try {
            parsed = JSON.parse(event.data) as ResponseStreamEvent
          } catch {
            continue
          }

          const anthropicEvents = translateResponsesStreamEvent(parsed, streamState)
          for (const ae of anthropicEvents) {
            sendEvent(ae)
          }
        }
      } catch (error) {
        consola.error("Responses stream error:", error)
        sendEvent(buildErrorEvent("Stream translation error"))
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
