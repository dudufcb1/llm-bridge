import { type AnthropicResponse } from "./anthropic-types"

export function mapOpenAIStopReasonToAnthropic(
  finishReason: "stop" | "length" | "tool_calls" | "content_filter" | null,
): AnthropicResponse["stop_reason"] {
  if (finishReason === null) {
    return null
  }
  const stopReasonMap = {
    stop: "end_turn",
    length: "max_tokens",
    tool_calls: "tool_use",
    content_filter: "end_turn",
  } as const
  return stopReasonMap[finishReason]
}

export function mapAnthropicStopReasonToOpenAI(
  stopReason: AnthropicResponse["stop_reason"],
): "stop" | "length" | "tool_calls" | "content_filter" | null {
  if (stopReason === null) {
    return null
  }
  const finishReasonMap: Record<string, "stop" | "length" | "tool_calls"> = {
    end_turn: "stop",
    max_tokens: "length",
    tool_use: "tool_calls",
    stop_sequence: "stop",
    pause_turn: "stop",
    refusal: "stop",
  }
  return finishReasonMap[stopReason] ?? "stop"
}
