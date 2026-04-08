// Responses API stream events -> Anthropic stream events (extracted from cpn8n, Copilot-decoupled)

import type {
  ResponseCompletedEvent,
  ResponseCreatedEvent,
  ResponseErrorEvent,
  ResponseFailedEvent,
  ResponseFunctionCallArgumentsDeltaEvent,
  ResponseFunctionCallArgumentsDoneEvent,
  ResponseIncompleteEvent,
  ResponseOutputItemAddedEvent,
  ResponseOutputItemDoneEvent,
  ResponseReasoningSummaryTextDeltaEvent,
  ResponseReasoningSummaryTextDoneEvent,
  ResponsesResult,
  ResponseStreamEvent,
  ResponseTextDeltaEvent,
  ResponseTextDoneEvent,
  ResponsesStreamState,
} from "~/lib/responses-types"

import type { AnthropicStreamEventData } from "~/lib/anthropic-types"
import { translateResponsesToAnthropic } from "./translation"

const MAX_CONSECUTIVE_FUNCTION_CALL_WHITESPACE = 20

export const createResponsesStreamState = (): ResponsesStreamState => ({
  messageStartSent: false,
  messageCompleted: false,
  nextContentBlockIndex: 0,
  blockIndexByKey: new Map(),
  openBlocks: new Set(),
  blockHasDelta: new Set(),
  functionCallStateByOutputIndex: new Map(),
})

export const translateResponsesStreamEvent = (
  rawEvent: ResponseStreamEvent,
  state: ResponsesStreamState,
): Array<AnthropicStreamEventData> => {
  switch (rawEvent.type) {
    case "response.created":
      return handleResponseCreated(rawEvent, state)
    case "response.output_item.added":
      return handleOutputItemAdded(rawEvent, state)
    case "response.reasoning_summary_text.delta":
      return handleReasoningSummaryTextDelta(rawEvent, state)
    case "response.output_text.delta":
      return handleOutputTextDelta(rawEvent, state)
    case "response.reasoning_summary_text.done":
      return handleReasoningSummaryTextDone(rawEvent, state)
    case "response.output_text.done":
      return handleOutputTextDone(rawEvent, state)
    case "response.output_item.done":
      return handleOutputItemDone(rawEvent, state)
    case "response.function_call_arguments.delta":
      return handleFunctionCallArgumentsDelta(rawEvent, state)
    case "response.function_call_arguments.done":
      return handleFunctionCallArgumentsDone(rawEvent, state)
    case "response.completed":
    case "response.incomplete":
      return handleResponseCompleted(rawEvent, state)
    case "response.failed":
      return handleResponseFailed(rawEvent, state)
    case "error":
      return handleErrorEvent(rawEvent, state)
    default:
      return []
  }
}

export const buildErrorEvent = (message: string): AnthropicStreamEventData => ({
  type: "error",
  error: { type: "api_error", message },
})

// Handlers

const handleResponseCreated = (
  rawEvent: ResponseCreatedEvent,
  state: ResponsesStreamState,
): Array<AnthropicStreamEventData> => {
  return messageStart(state, rawEvent.response)
}

const handleOutputItemAdded = (
  rawEvent: ResponseOutputItemAddedEvent,
  state: ResponsesStreamState,
): Array<AnthropicStreamEventData> => {
  const events: Array<AnthropicStreamEventData> = []
  const item = rawEvent.item
  if (item.type !== "function_call") return events

  const outputIndex = rawEvent.output_index
  const blockIndex = openFunctionCallBlock(state, {
    outputIndex,
    toolCallId: item.call_id,
    name: item.name,
    events,
  })

  const initialArguments = item.arguments
  if (initialArguments && initialArguments.length > 0) {
    events.push({
      type: "content_block_delta",
      index: blockIndex,
      delta: { type: "input_json_delta", partial_json: initialArguments },
    })
    state.blockHasDelta.add(blockIndex)
  }

  return events
}

const handleOutputItemDone = (
  rawEvent: ResponseOutputItemDoneEvent,
  state: ResponsesStreamState,
): Array<AnthropicStreamEventData> => {
  const events: Array<AnthropicStreamEventData> = []
  const item = rawEvent.item
  if (item.type !== "reasoning") return events

  const outputIndex = rawEvent.output_index
  const blockIndex = openThinkingBlockIfNeeded(state, outputIndex, events)
  const signature = (item.encrypted_content ?? "") + "@" + item.id

  events.push({
    type: "content_block_delta",
    index: blockIndex,
    delta: { type: "signature_delta", signature },
  })
  state.blockHasDelta.add(blockIndex)

  return events
}

const handleFunctionCallArgumentsDelta = (
  rawEvent: ResponseFunctionCallArgumentsDeltaEvent,
  state: ResponsesStreamState,
): Array<AnthropicStreamEventData> => {
  const events: Array<AnthropicStreamEventData> = []
  const { output_index: outputIndex, delta: deltaText } = rawEvent

  if (!deltaText) return events

  const blockIndex = openFunctionCallBlock(state, { outputIndex, events })

  const functionCallState = state.functionCallStateByOutputIndex.get(outputIndex)
  if (!functionCallState) {
    closeAllOpenBlocks(state, events)
    state.messageCompleted = true
    events.push(buildErrorEvent("Received function call arguments delta without an open tool call block."))
    return events
  }

  // Validate whitespace (prevents infinite whitespace from reasoning models)
  let count = functionCallState.consecutiveWhitespaceCount
  for (const char of deltaText) {
    if (char === "\r" || char === "\n" || char === "\t") {
      count += 1
      if (count > MAX_CONSECUTIVE_FUNCTION_CALL_WHITESPACE) {
        closeAllOpenBlocks(state, events)
        state.messageCompleted = true
        events.push(buildErrorEvent("Function call arguments exceeded max consecutive whitespace."))
        return events
      }
    } else if (char !== " ") {
      count = 0
    }
  }
  functionCallState.consecutiveWhitespaceCount = count

  events.push({
    type: "content_block_delta",
    index: blockIndex,
    delta: { type: "input_json_delta", partial_json: deltaText },
  })
  state.blockHasDelta.add(blockIndex)

  return events
}

const handleFunctionCallArgumentsDone = (
  rawEvent: ResponseFunctionCallArgumentsDoneEvent,
  state: ResponsesStreamState,
): Array<AnthropicStreamEventData> => {
  const events: Array<AnthropicStreamEventData> = []
  const { output_index: outputIndex } = rawEvent
  const blockIndex = openFunctionCallBlock(state, { outputIndex, events })

  const finalArguments = typeof rawEvent.arguments === "string" ? rawEvent.arguments : undefined
  if (!state.blockHasDelta.has(blockIndex) && finalArguments) {
    events.push({
      type: "content_block_delta",
      index: blockIndex,
      delta: { type: "input_json_delta", partial_json: finalArguments },
    })
    state.blockHasDelta.add(blockIndex)
  }

  state.functionCallStateByOutputIndex.delete(outputIndex)
  return events
}

const handleOutputTextDelta = (
  rawEvent: ResponseTextDeltaEvent,
  state: ResponsesStreamState,
): Array<AnthropicStreamEventData> => {
  const events: Array<AnthropicStreamEventData> = []
  const { output_index: outputIndex, content_index: contentIndex, delta: deltaText } = rawEvent

  if (!deltaText) return events

  const blockIndex = openTextBlockIfNeeded(state, { outputIndex, contentIndex, events })
  events.push({
    type: "content_block_delta",
    index: blockIndex,
    delta: { type: "text_delta", text: deltaText },
  })
  state.blockHasDelta.add(blockIndex)

  return events
}

const handleReasoningSummaryTextDelta = (
  rawEvent: ResponseReasoningSummaryTextDeltaEvent,
  state: ResponsesStreamState,
): Array<AnthropicStreamEventData> => {
  const events: Array<AnthropicStreamEventData> = []
  const blockIndex = openThinkingBlockIfNeeded(state, rawEvent.output_index, events)

  events.push({
    type: "content_block_delta",
    index: blockIndex,
    delta: { type: "thinking_delta", thinking: rawEvent.delta },
  })
  state.blockHasDelta.add(blockIndex)

  return events
}

const handleReasoningSummaryTextDone = (
  rawEvent: ResponseReasoningSummaryTextDoneEvent,
  state: ResponsesStreamState,
): Array<AnthropicStreamEventData> => {
  const events: Array<AnthropicStreamEventData> = []
  const blockIndex = openThinkingBlockIfNeeded(state, rawEvent.output_index, events)

  if (rawEvent.text && !state.blockHasDelta.has(blockIndex)) {
    events.push({
      type: "content_block_delta",
      index: blockIndex,
      delta: { type: "thinking_delta", thinking: rawEvent.text },
    })
  }

  return events
}

const handleOutputTextDone = (
  rawEvent: ResponseTextDoneEvent,
  state: ResponsesStreamState,
): Array<AnthropicStreamEventData> => {
  const events: Array<AnthropicStreamEventData> = []
  const blockIndex = openTextBlockIfNeeded(state, {
    outputIndex: rawEvent.output_index,
    contentIndex: rawEvent.content_index,
    events,
  })

  if (rawEvent.text && !state.blockHasDelta.has(blockIndex)) {
    events.push({
      type: "content_block_delta",
      index: blockIndex,
      delta: { type: "text_delta", text: rawEvent.text },
    })
  }

  return events
}

const handleResponseCompleted = (
  rawEvent: ResponseCompletedEvent | ResponseIncompleteEvent,
  state: ResponsesStreamState,
): Array<AnthropicStreamEventData> => {
  const events: Array<AnthropicStreamEventData> = []
  closeAllOpenBlocks(state, events)

  const anthropic = translateResponsesToAnthropic(rawEvent.response)
  events.push(
    {
      type: "message_delta",
      delta: { stop_reason: anthropic.stop_reason, stop_sequence: anthropic.stop_sequence },
      usage: anthropic.usage,
    },
    { type: "message_stop" },
  )
  state.messageCompleted = true
  return events
}

const handleResponseFailed = (
  rawEvent: ResponseFailedEvent,
  state: ResponsesStreamState,
): Array<AnthropicStreamEventData> => {
  const events: Array<AnthropicStreamEventData> = []
  closeAllOpenBlocks(state, events)
  events.push(buildErrorEvent(rawEvent.response.error?.message ?? "Response failed"))
  state.messageCompleted = true
  return events
}

const handleErrorEvent = (
  rawEvent: ResponseErrorEvent,
  state: ResponsesStreamState,
): Array<AnthropicStreamEventData> => {
  state.messageCompleted = true
  return [buildErrorEvent(rawEvent.message || "An unexpected error occurred during streaming.")]
}

// Block management helpers

const messageStart = (
  state: ResponsesStreamState,
  response: ResponsesResult,
): Array<AnthropicStreamEventData> => {
  state.messageStartSent = true
  const inputCachedTokens = response.usage?.input_tokens_details?.cached_tokens
  const inputTokens = (response.usage?.input_tokens ?? 0) - (inputCachedTokens ?? 0)
  return [{
    type: "message_start",
    message: {
      id: response.id,
      type: "message",
      role: "assistant",
      content: [],
      model: response.model,
      stop_reason: null,
      stop_sequence: null,
      usage: {
        input_tokens: inputTokens,
        output_tokens: 0,
        cache_read_input_tokens: inputCachedTokens ?? 0,
      },
    },
  }]
}

const getBlockKey = (outputIndex: number, contentIndex: number): string =>
  `${outputIndex}:${contentIndex}`

const openTextBlockIfNeeded = (
  state: ResponsesStreamState,
  params: { outputIndex: number; contentIndex: number; events: Array<AnthropicStreamEventData> },
): number => {
  const key = getBlockKey(params.outputIndex, params.contentIndex)
  let blockIndex = state.blockIndexByKey.get(key)

  if (blockIndex === undefined) {
    blockIndex = state.nextContentBlockIndex++
    state.blockIndexByKey.set(key, blockIndex)
  }

  if (!state.openBlocks.has(blockIndex)) {
    closeOpenBlocks(state, params.events)
    params.events.push({
      type: "content_block_start",
      index: blockIndex,
      content_block: { type: "text", text: "" },
    })
    state.openBlocks.add(blockIndex)
  }

  return blockIndex
}

const openThinkingBlockIfNeeded = (
  state: ResponsesStreamState,
  outputIndex: number,
  events: Array<AnthropicStreamEventData>,
): number => {
  const key = getBlockKey(outputIndex, 0) // combine all summary indices into one block
  let blockIndex = state.blockIndexByKey.get(key)

  if (blockIndex === undefined) {
    blockIndex = state.nextContentBlockIndex++
    state.blockIndexByKey.set(key, blockIndex)
  }

  if (!state.openBlocks.has(blockIndex)) {
    closeOpenBlocks(state, events)
    events.push({
      type: "content_block_start",
      index: blockIndex,
      content_block: { type: "thinking", thinking: "" },
    })
    state.openBlocks.add(blockIndex)
  }

  return blockIndex
}

const openFunctionCallBlock = (
  state: ResponsesStreamState,
  params: {
    outputIndex: number
    toolCallId?: string
    name?: string
    events: Array<AnthropicStreamEventData>
  },
): number => {
  let functionCallState = state.functionCallStateByOutputIndex.get(params.outputIndex)

  if (!functionCallState) {
    const blockIndex = state.nextContentBlockIndex++
    functionCallState = {
      blockIndex,
      toolCallId: params.toolCallId ?? `tool_call_${blockIndex}`,
      name: params.name ?? "function",
      consecutiveWhitespaceCount: 0,
    }
    state.functionCallStateByOutputIndex.set(params.outputIndex, functionCallState)
  }

  const { blockIndex } = functionCallState

  if (!state.openBlocks.has(blockIndex)) {
    closeOpenBlocks(state, params.events)
    params.events.push({
      type: "content_block_start",
      index: blockIndex,
      content_block: {
        type: "tool_use",
        id: functionCallState.toolCallId,
        name: functionCallState.name,
        input: {},
      },
    })
    state.openBlocks.add(blockIndex)
  }

  return blockIndex
}

const closeOpenBlocks = (
  state: ResponsesStreamState,
  events: Array<AnthropicStreamEventData>,
) => {
  for (const blockIndex of state.openBlocks) {
    events.push({ type: "content_block_stop", index: blockIndex })
  }
  state.openBlocks.clear()
  state.blockHasDelta.clear()
}

const closeAllOpenBlocks = (
  state: ResponsesStreamState,
  events: Array<AnthropicStreamEventData>,
) => {
  closeOpenBlocks(state, events)
  state.functionCallStateByOutputIndex.clear()
}
