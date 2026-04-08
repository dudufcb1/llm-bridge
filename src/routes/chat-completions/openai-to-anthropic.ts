// OpenAI Chat Completions -> Anthropic Messages translation

import type {
  ChatCompletionsPayload,
  ChatCompletionResponse,
  Message,
  ToolCall,
} from "~/lib/openai-types"

import type {
  AnthropicMessagesPayload,
  AnthropicMessage,
  AnthropicResponse,
  AnthropicTool,
  AnthropicUserContentBlock,
  AnthropicAssistantContentBlock,
  AnthropicTextBlock,
  AnthropicToolUseBlock,
  AnthropicThinkingBlock,
} from "~/lib/anthropic-types"

import { mapAnthropicStopReasonToOpenAI } from "~/lib/utils"

// Payload translation: OpenAI -> Anthropic

export function translateToAnthropic(
  payload: ChatCompletionsPayload,
): AnthropicMessagesPayload {
  const { systemPrompt, messages } = extractSystemAndMessages(payload.messages)

  return {
    model: payload.model,
    messages,
    max_tokens: payload.max_tokens ?? 4096,
    system: systemPrompt,
    stop_sequences: typeof payload.stop === "string" ? [payload.stop] : payload.stop ?? undefined,
    stream: payload.stream ?? undefined,
    temperature: payload.temperature ?? undefined,
    top_p: payload.top_p ?? undefined,
    metadata: payload.user ? { user_id: payload.user } : undefined,
    tools: translateOpenAIToolsToAnthropic(payload.tools),
    tool_choice: translateOpenAIToolChoiceToAnthropic(payload.tool_choice),
    thinking: payload.thinking_budget ? { type: "enabled", budget_tokens: payload.thinking_budget } : undefined,
  }
}

function extractSystemAndMessages(
  openaiMessages: Array<Message>,
): { systemPrompt: string | undefined; messages: Array<AnthropicMessage> } {
  let systemPrompt: string | undefined
  const anthropicMessages: Array<AnthropicMessage> = []

  // Collect pending tool results to merge into next user message
  let pendingToolResults: Array<AnthropicUserContentBlock> = []

  for (const msg of openaiMessages) {
    if (msg.role === "system" || msg.role === "developer") {
      const text = typeof msg.content === "string" ? msg.content : ""
      systemPrompt = systemPrompt ? `${systemPrompt}\n\n${text}` : text
      continue
    }

    if (msg.role === "tool") {
      pendingToolResults.push({
        type: "tool_result",
        tool_use_id: msg.tool_call_id || "",
        content: typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content),
      })
      continue
    }

    if (msg.role === "user") {
      const userContent: Array<AnthropicUserContentBlock> = []

      // Flush pending tool results first
      if (pendingToolResults.length > 0) {
        userContent.push(...pendingToolResults)
        pendingToolResults = []
      }

      if (typeof msg.content === "string") {
        userContent.push({ type: "text", text: msg.content })
      } else if (Array.isArray(msg.content)) {
        for (const part of msg.content) {
          if (part.type === "text") {
            userContent.push({ type: "text", text: part.text })
          } else if (part.type === "image_url") {
            const url = part.image_url.url
            const match = url.match(/^data:([^;]+);base64,(.+)$/)
            if (match) {
              userContent.push({
                type: "image",
                source: {
                  type: "base64",
                  media_type: match[1] as "image/jpeg" | "image/png" | "image/gif" | "image/webp",
                  data: match[2],
                },
              })
            }
          }
        }
      }

      anthropicMessages.push({ role: "user", content: userContent })
      continue
    }

    if (msg.role === "assistant") {
      // Flush pending tool results as a user message before assistant
      if (pendingToolResults.length > 0) {
        anthropicMessages.push({ role: "user", content: pendingToolResults })
        pendingToolResults = []
      }

      const assistantContent: Array<AnthropicAssistantContentBlock> = []

      if (msg.reasoning_text) {
        assistantContent.push({
          type: "thinking",
          thinking: msg.reasoning_text,
          signature: msg.reasoning_opaque || "",
        })
      }

      if (typeof msg.content === "string" && msg.content.length > 0) {
        assistantContent.push({ type: "text", text: msg.content })
      } else if (Array.isArray(msg.content)) {
        for (const part of msg.content) {
          if (part.type === "text") {
            assistantContent.push({ type: "text", text: part.text })
          }
        }
      }

      if (msg.tool_calls) {
        for (const tc of msg.tool_calls) {
          assistantContent.push({
            type: "tool_use",
            id: tc.id,
            name: tc.function.name,
            input: JSON.parse(tc.function.arguments) as Record<string, unknown>,
          })
        }
      }

      anthropicMessages.push({
        role: "assistant",
        content: assistantContent.length > 0 ? assistantContent : [{ type: "text", text: "" }],
      })
      continue
    }
  }

  // Flush any remaining tool results
  if (pendingToolResults.length > 0) {
    anthropicMessages.push({ role: "user", content: pendingToolResults })
  }

  return { systemPrompt, messages: anthropicMessages }
}

function translateOpenAIToolsToAnthropic(
  tools: ChatCompletionsPayload["tools"],
): Array<AnthropicTool> | undefined {
  if (!tools) {
    return undefined
  }
  return tools.map((tool) => ({
    name: tool.function.name,
    description: tool.function.description,
    input_schema: tool.function.parameters,
  }))
}

function translateOpenAIToolChoiceToAnthropic(
  toolChoice: ChatCompletionsPayload["tool_choice"],
): AnthropicMessagesPayload["tool_choice"] {
  if (!toolChoice) {
    return undefined
  }

  if (toolChoice === "auto") {
    return { type: "auto" }
  }
  if (toolChoice === "required") {
    return { type: "any" }
  }
  if (toolChoice === "none") {
    return { type: "none" }
  }
  if (typeof toolChoice === "object" && toolChoice.type === "function") {
    return { type: "tool", name: toolChoice.function.name }
  }
  return undefined
}

// Response translation: Anthropic -> OpenAI

export function translateToOpenAI(
  response: AnthropicResponse,
): ChatCompletionResponse {
  const textContent = response.content
    .filter((b): b is AnthropicTextBlock => b.type === "text")
    .map((b) => b.text)
    .join("")

  const toolCalls: Array<ToolCall> = response.content
    .filter((b): b is AnthropicToolUseBlock => b.type === "tool_use")
    .map((b) => ({
      id: b.id,
      type: "function",
      function: {
        name: b.name,
        arguments: JSON.stringify(b.input),
      },
    }))

  const thinkingBlocks = response.content.filter(
    (b): b is AnthropicThinkingBlock => b.type === "thinking",
  )
  const reasoningText = thinkingBlocks
    .filter((b) => b.thinking.length > 0)
    .map((b) => b.thinking)
    .join("\n\n") || null
  const reasoningOpaque = thinkingBlocks.find((b) => b.signature.length > 0)?.signature ?? null

  return {
    id: response.id,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: response.model,
    choices: [
      {
        index: 0,
        message: {
          role: "assistant",
          content: textContent || null,
          ...(reasoningText && { reasoning_text: reasoningText }),
          ...(reasoningOpaque && { reasoning_opaque: reasoningOpaque }),
          ...(toolCalls.length > 0 && { tool_calls: toolCalls }),
        },
        logprobs: null,
        finish_reason: mapAnthropicStopReasonToOpenAI(response.stop_reason) || "stop",
      },
    ],
    usage: {
      prompt_tokens: response.usage.input_tokens,
      completion_tokens: response.usage.output_tokens,
      total_tokens: response.usage.input_tokens + response.usage.output_tokens,
      ...(response.usage.cache_read_input_tokens !== undefined && {
        prompt_tokens_details: {
          cached_tokens: response.usage.cache_read_input_tokens,
        },
      }),
    },
  }
}
