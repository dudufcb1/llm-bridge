import { describe, expect, test } from "bun:test"
import { mapOpenAIStopReasonToAnthropic, mapAnthropicStopReasonToOpenAI } from "~/lib/utils"

describe("mapOpenAIStopReasonToAnthropic", () => {
  test("maps stop -> end_turn", () => {
    expect(mapOpenAIStopReasonToAnthropic("stop")).toBe("end_turn")
  })

  test("maps length -> max_tokens", () => {
    expect(mapOpenAIStopReasonToAnthropic("length")).toBe("max_tokens")
  })

  test("maps tool_calls -> tool_use", () => {
    expect(mapOpenAIStopReasonToAnthropic("tool_calls")).toBe("tool_use")
  })

  test("maps content_filter -> end_turn", () => {
    expect(mapOpenAIStopReasonToAnthropic("content_filter")).toBe("end_turn")
  })

  test("maps null -> null", () => {
    expect(mapOpenAIStopReasonToAnthropic(null)).toBeNull()
  })
})

describe("mapAnthropicStopReasonToOpenAI", () => {
  test("maps end_turn -> stop", () => {
    expect(mapAnthropicStopReasonToOpenAI("end_turn")).toBe("stop")
  })

  test("maps max_tokens -> length", () => {
    expect(mapAnthropicStopReasonToOpenAI("max_tokens")).toBe("length")
  })

  test("maps tool_use -> tool_calls", () => {
    expect(mapAnthropicStopReasonToOpenAI("tool_use")).toBe("tool_calls")
  })

  test("maps stop_sequence -> stop", () => {
    expect(mapAnthropicStopReasonToOpenAI("stop_sequence")).toBe("stop")
  })

  test("maps pause_turn -> stop", () => {
    expect(mapAnthropicStopReasonToOpenAI("pause_turn")).toBe("stop")
  })

  test("maps refusal -> stop", () => {
    expect(mapAnthropicStopReasonToOpenAI("refusal")).toBe("stop")
  })

  test("maps null -> null", () => {
    expect(mapAnthropicStopReasonToOpenAI(null)).toBeNull()
  })
})
