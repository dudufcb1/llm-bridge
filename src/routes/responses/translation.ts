// Anthropic Messages <-> OpenAI Responses API translation (extracted from cpn8n, Copilot-decoupled)

import type {
  ResponsesPayload,
  ResponseInputContent,
  ResponseInputImage,
  ResponseInputItem,
  ResponseInputMessage,
  ResponseInputText,
  ResponsesResult,
  ResponseOutputContentBlock,
  ResponseOutputFunctionCall,
  ResponseOutputItem,
  ResponseOutputReasoning,
  ResponseOutputText,
  ResponseOutputRefusal,
  ResponseReasoningBlock,
  ResponseFunctionToolCallItem,
  ResponseFunctionCallOutputItem,
  ResponseTool,
  ToolChoiceFunction,
  ToolChoiceOptions,
} from "~/lib/responses-types"

import type {
  AnthropicAssistantContentBlock,
  AnthropicAssistantMessage,
  AnthropicResponse,
  AnthropicImageBlock,
  AnthropicMessage,
  AnthropicMessagesPayload,
  AnthropicTextBlock,
  AnthropicThinkingBlock,
  AnthropicTool,
  AnthropicToolResultBlock,
  AnthropicToolUseBlock,
  AnthropicUserContentBlock,
  AnthropicUserMessage,
} from "~/lib/anthropic-types"

import consola from "consola"

const MESSAGE_TYPE = "message"

// Anthropic -> Responses payload

export const translateAnthropicToResponsesPayload = (
  payload: AnthropicMessagesPayload,
): ResponsesPayload => {
  const input: Array<ResponseInputItem> = []

  for (const message of payload.messages) {
    input.push(...translateMessage(message))
  }

  const translatedTools = convertAnthropicTools(payload.tools)
  const toolChoice = convertAnthropicToolChoice(payload.tool_choice)

  const result: ResponsesPayload = {
    model: payload.model,
    input,
    instructions: translateSystemPrompt(payload.system),
    temperature: payload.temperature ?? undefined,
    top_p: payload.top_p ?? undefined,
    max_output_tokens: payload.max_tokens,
    tools: translatedTools,
    tool_choice: toolChoice,
    metadata: payload.metadata ? { ...payload.metadata } as Record<string, string> : null,
    stream: payload.stream ?? null,
    store: false,
    parallel_tool_calls: true,
  }

  // Only add reasoning config for models that support it
  if (payload.thinking) {
    result.reasoning = { effort: "high", summary: "detailed" }
    result.include = ["reasoning.encrypted_content"]
  }

  return result
}

const translateMessage = (
  message: AnthropicMessage,
): Array<ResponseInputItem> => {
  if (message.role === "user") {
    return translateUserMessage(message)
  }
  return translateAssistantMessage(message)
}

const translateUserMessage = (
  message: AnthropicUserMessage,
): Array<ResponseInputItem> => {
  if (typeof message.content === "string") {
    return [createMessage("user", message.content)]
  }

  if (!Array.isArray(message.content)) {
    return []
  }

  const items: Array<ResponseInputItem> = []
  const pendingContent: Array<ResponseInputContent> = []

  for (const block of message.content) {
    if (block.type === "tool_result") {
      flushPendingContent("user", pendingContent, items)
      items.push(createFunctionCallOutput(block))
      continue
    }

    const converted = translateUserContentBlock(block)
    if (converted) {
      pendingContent.push(converted)
    }
  }

  flushPendingContent("user", pendingContent, items)
  return items
}

const translateAssistantMessage = (
  message: AnthropicAssistantMessage,
): Array<ResponseInputItem> => {
  if (typeof message.content === "string") {
    return [createMessage("assistant", message.content)]
  }

  if (!Array.isArray(message.content)) {
    return []
  }

  const items: Array<ResponseInputItem> = []
  const pendingContent: Array<ResponseInputContent> = []

  for (const block of message.content) {
    if (block.type === "tool_use") {
      flushPendingContent("assistant", pendingContent, items)
      items.push(createFunctionToolCall(block))
      continue
    }

    if (block.type === "thinking" && block.signature && block.signature.length > 0) {
      flushPendingContent("assistant", pendingContent, items)
      items.push(createReasoningContent(block))
      continue
    }

    const converted = translateAssistantContentBlock(block)
    if (converted) {
      pendingContent.push(converted)
    }
  }

  flushPendingContent("assistant", pendingContent, items)
  return items
}

const translateUserContentBlock = (
  block: AnthropicUserContentBlock,
): ResponseInputContent | undefined => {
  switch (block.type) {
    case "text":
      return { type: "input_text", text: block.text }
    case "image":
      return createImageContent(block)
    default:
      return undefined
  }
}

const translateAssistantContentBlock = (
  block: AnthropicAssistantContentBlock,
): ResponseInputContent | undefined => {
  if (block.type === "text") {
    return { type: "output_text", text: block.text }
  }
  return undefined
}

const flushPendingContent = (
  role: ResponseInputMessage["role"],
  pendingContent: Array<ResponseInputContent>,
  target: Array<ResponseInputItem>,
) => {
  if (pendingContent.length === 0) return
  target.push(createMessage(role, [...pendingContent]))
  pendingContent.length = 0
}

const createMessage = (
  role: ResponseInputMessage["role"],
  content: string | Array<ResponseInputContent>,
): ResponseInputMessage => ({
  type: MESSAGE_TYPE,
  role,
  content,
})

const createImageContent = (block: AnthropicImageBlock): ResponseInputImage => ({
  type: "input_image",
  image_url: `data:${block.source.media_type};base64,${block.source.data}`,
  detail: "auto",
})

const createReasoningContent = (block: AnthropicThinkingBlock) => {
  // Signature may contain "@" separator from Responses API round-trips
  const atIndex = block.signature.indexOf("@")
  if (atIndex >= 0) {
    return {
      type: "reasoning" as const,
      id: block.signature.slice(atIndex + 1),
      summary: [{ type: "summary_text" as const, text: block.thinking }],
      encrypted_content: block.signature.slice(0, atIndex),
    }
  }
  // Plain signature (from Chat Completions round-trip)
  return {
    type: "reasoning" as const,
    summary: [{ type: "summary_text" as const, text: block.thinking }],
    encrypted_content: block.signature,
  }
}

const createFunctionToolCall = (block: AnthropicToolUseBlock): ResponseFunctionToolCallItem => ({
  type: "function_call",
  call_id: block.id,
  name: block.name,
  arguments: JSON.stringify(block.input),
  status: "completed",
})

const createFunctionCallOutput = (block: AnthropicToolResultBlock): ResponseFunctionCallOutputItem => ({
  type: "function_call_output",
  call_id: block.tool_use_id,
  output: block.content,
  status: block.is_error ? "incomplete" : "completed",
})

const translateSystemPrompt = (
  system: string | Array<AnthropicTextBlock> | undefined,
): string | null => {
  if (!system) return null
  if (typeof system === "string") return system
  const text = system.map((b) => b.text).join(" ")
  return text.length > 0 ? text : null
}

const convertAnthropicTools = (
  tools: Array<AnthropicTool> | undefined,
): Array<ResponseTool> | null => {
  if (!tools || tools.length === 0) return null
  return tools.map((tool) => ({
    type: "function",
    name: tool.name,
    parameters: tool.input_schema,
    strict: false,
    ...(tool.description ? { description: tool.description } : {}),
  }))
}

const convertAnthropicToolChoice = (
  choice: AnthropicMessagesPayload["tool_choice"],
): ToolChoiceOptions | ToolChoiceFunction => {
  if (!choice) return "auto"
  switch (choice.type) {
    case "auto": return "auto"
    case "any": return "required"
    case "tool": return choice.name ? { type: "function", name: choice.name } : "auto"
    case "none": return "none"
    default: return "auto"
  }
}

// Responses result -> Anthropic response

export const translateResponsesToAnthropic = (
  response: ResponsesResult,
): AnthropicResponse => {
  const contentBlocks = mapOutputToAnthropicContent(response.output)
  let anthropicContent: Array<AnthropicAssistantContentBlock> = contentBlocks

  if (contentBlocks.length === 0 && response.output_text) {
    anthropicContent = [{ type: "text", text: response.output_text }]
  }

  return {
    id: response.id,
    type: "message",
    role: "assistant",
    content: anthropicContent,
    model: response.model,
    stop_reason: mapResponsesStopReason(response),
    stop_sequence: null,
    usage: mapResponsesUsage(response),
  }
}

const mapOutputToAnthropicContent = (
  output: Array<ResponseOutputItem>,
): Array<AnthropicAssistantContentBlock> => {
  const contentBlocks: Array<AnthropicAssistantContentBlock> = []

  for (const item of output) {
    switch (item.type) {
      case "reasoning": {
        const thinkingText = extractReasoningText(item)
        if (thinkingText.length > 0 || item.encrypted_content) {
          contentBlocks.push({
            type: "thinking",
            thinking: thinkingText,
            signature: (item.encrypted_content ?? "") + "@" + item.id,
          })
        }
        break
      }
      case "function_call": {
        const toolUseBlock = createToolUseContentBlock(item)
        if (toolUseBlock) contentBlocks.push(toolUseBlock)
        break
      }
      case "message": {
        const combinedText = combineMessageTextContent(item.content)
        if (combinedText.length > 0) {
          contentBlocks.push({ type: "text", text: combinedText })
        }
        break
      }
    }
  }

  return contentBlocks
}

const combineMessageTextContent = (
  content: Array<ResponseOutputContentBlock> | undefined,
): string => {
  if (!Array.isArray(content)) return ""

  let aggregated = ""
  for (const block of content) {
    if (isResponseOutputText(block)) {
      aggregated += block.text
    } else if (isResponseOutputRefusal(block)) {
      aggregated += block.refusal
    } else if (typeof (block as { text?: unknown }).text === "string") {
      aggregated += (block as { text: string }).text
    }
  }
  return aggregated
}

const extractReasoningText = (item: ResponseOutputReasoning): string => {
  const segments: Array<string> = []
  if (Array.isArray(item.summary)) {
    for (const block of item.summary) {
      if (typeof block.text === "string") segments.push(block.text)
    }
  }
  return segments.join("").trim()
}

const createToolUseContentBlock = (
  call: ResponseOutputFunctionCall,
): AnthropicToolUseBlock | null => {
  if (!call.name || !call.call_id) return null

  let input: Record<string, unknown> = {}
  if (typeof call.arguments === "string" && call.arguments.trim().length > 0) {
    try {
      const parsed = JSON.parse(call.arguments)
      input = (parsed && typeof parsed === "object" && !Array.isArray(parsed))
        ? parsed as Record<string, unknown>
        : { arguments: parsed }
    } catch {
      input = { raw_arguments: call.arguments }
    }
  }

  return {
    type: "tool_use",
    id: call.call_id,
    name: call.name,
    input,
  }
}

const mapResponsesStopReason = (
  response: ResponsesResult,
): AnthropicResponse["stop_reason"] => {
  if (response.status === "completed") {
    if (response.output.some((item) => item.type === "function_call")) return "tool_use"
    return "end_turn"
  }
  if (response.status === "incomplete") {
    if (response.incomplete_details?.reason === "max_output_tokens") return "max_tokens"
    if (response.incomplete_details?.reason === "content_filter") return "end_turn"
  }
  return null
}

const mapResponsesUsage = (
  response: ResponsesResult,
): AnthropicResponse["usage"] => {
  const inputTokens = response.usage?.input_tokens ?? 0
  const outputTokens = response.usage?.output_tokens ?? 0
  const inputCachedTokens = response.usage?.input_tokens_details?.cached_tokens

  return {
    input_tokens: inputTokens - (inputCachedTokens ?? 0),
    output_tokens: outputTokens,
    ...(inputCachedTokens !== undefined && {
      cache_read_input_tokens: inputCachedTokens,
    }),
  }
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null

const isResponseOutputText = (block: ResponseOutputContentBlock): block is ResponseOutputText =>
  isRecord(block) && (block as { type?: unknown }).type === "output_text"

const isResponseOutputRefusal = (block: ResponseOutputContentBlock): block is ResponseOutputRefusal =>
  isRecord(block) && (block as { type?: unknown }).type === "refusal"
