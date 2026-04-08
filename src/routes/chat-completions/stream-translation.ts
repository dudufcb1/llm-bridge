// Anthropic streaming events -> OpenAI streaming chunks translation

import type { ChatCompletionChunk } from "~/lib/openai-types"
import { mapAnthropicStopReasonToOpenAI } from "~/lib/utils"

// State for tracking Anthropic -> OpenAI stream conversion
export interface OpenAIStreamState {
  id: string
  model: string
  created: number
  currentToolCallIndex: number
  // Track tool_use block indices to tool_call indices
  toolBlockMap: { [anthropicBlockIndex: number]: number }
}

export function createOpenAIStreamState(): OpenAIStreamState {
  return {
    id: `chatcmpl-${crypto.randomUUID()}`,
    model: "",
    created: Math.floor(Date.now() / 1000),
    currentToolCallIndex: 0,
    toolBlockMap: {},
  }
}

export function translateAnthropicEventToOpenAIChunks(
  event: { event: string; data: string },
  state: OpenAIStreamState,
): Array<ChatCompletionChunk> {
  const chunks: Array<ChatCompletionChunk> = []

  let parsed: Record<string, unknown>
  try {
    parsed = JSON.parse(event.data)
  } catch {
    return chunks
  }

  const eventType = parsed.type as string

  switch (eventType) {
    case "message_start": {
      const message = parsed.message as Record<string, unknown>
      state.id = (message.id as string) || state.id
      state.model = (message.model as string) || state.model

      // Send initial role chunk
      chunks.push(makeChunk(state, {
        role: "assistant",
        content: "",
      }, null))
      break
    }

    case "content_block_start": {
      const block = parsed.content_block as Record<string, unknown>
      const index = parsed.index as number
      if (block.type === "tool_use") {
        const toolIndex = state.currentToolCallIndex++
        state.toolBlockMap[index] = toolIndex
        chunks.push(makeChunk(state, {
          tool_calls: [{
            index: toolIndex,
            id: block.id as string,
            type: "function",
            function: {
              name: block.name as string,
              arguments: "",
            },
          }],
        }, null))
      }
      break
    }

    case "content_block_delta": {
      const delta = parsed.delta as Record<string, unknown>
      const index = parsed.index as number

      if (delta.type === "text_delta") {
        chunks.push(makeChunk(state, {
          content: delta.text as string,
        }, null))
      } else if (delta.type === "input_json_delta") {
        const toolIndex = state.toolBlockMap[index]
        if (toolIndex !== undefined) {
          chunks.push(makeChunk(state, {
            tool_calls: [{
              index: toolIndex,
              function: {
                arguments: delta.partial_json as string,
              },
            }],
          }, null))
        }
      } else if (delta.type === "thinking_delta") {
        chunks.push(makeChunk(state, {
          reasoning_text: delta.thinking as string,
        }, null))
      } else if (delta.type === "signature_delta") {
        chunks.push(makeChunk(state, {
          reasoning_opaque: delta.signature as string,
        }, null))
      }
      break
    }

    case "message_delta": {
      const msgDelta = parsed.delta as Record<string, unknown>
      const stopReason = msgDelta.stop_reason as string | null
      const usage = parsed.usage as Record<string, number> | undefined

      const finishReason = stopReason
        ? mapAnthropicStopReasonToOpenAI(stopReason as Parameters<typeof mapAnthropicStopReasonToOpenAI>[0])
        : "stop"

      const chunk = makeChunk(state, {}, finishReason)
      if (usage) {
        chunk.usage = {
          prompt_tokens: 0,
          completion_tokens: usage.output_tokens ?? 0,
          total_tokens: usage.output_tokens ?? 0,
        }
      }
      chunks.push(chunk)
      break
    }

    case "message_stop": {
      // Terminal event, nothing to emit (client handles [DONE])
      break
    }
  }

  return chunks
}

function makeChunk(
  state: OpenAIStreamState,
  delta: Record<string, unknown>,
  finishReason: string | null,
): ChatCompletionChunk {
  return {
    id: state.id,
    object: "chat.completion.chunk",
    created: state.created,
    model: state.model,
    choices: [{
      index: 0,
      delta: delta as ChatCompletionChunk["choices"][0]["delta"],
      finish_reason: finishReason as ChatCompletionChunk["choices"][0]["finish_reason"],
      logprobs: null,
    }],
  }
}
