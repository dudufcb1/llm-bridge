import { describe, expect, test } from "bun:test"
import {
  translateAnthropicEventToOpenAIChunks,
  createOpenAIStreamState,
} from "~/routes/chat-completions/stream-translation"

function makeEvent(type: string, data: Record<string, unknown>) {
  return {
    event: type,
    data: JSON.stringify({ type, ...data }),
  }
}

describe("translateAnthropicEventToOpenAIChunks (Anthropic stream -> OpenAI chunks)", () => {
  test("emits role chunk on message_start", () => {
    const state = createOpenAIStreamState()
    const event = makeEvent("message_start", {
      message: {
        id: "msg_123",
        model: "claude-sonnet-4-20250514",
        role: "assistant",
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: 10, output_tokens: 0 },
      },
    })

    const chunks = translateAnthropicEventToOpenAIChunks(event, state)

    expect(chunks).toHaveLength(1)
    expect(chunks[0].choices[0].delta.role).toBe("assistant")
    expect(state.model).toBe("claude-sonnet-4-20250514")
  })

  test("emits content chunk on text_delta", () => {
    const state = createOpenAIStreamState()
    state.model = "claude-sonnet-4-20250514"

    const event = makeEvent("content_block_delta", {
      index: 0,
      delta: { type: "text_delta", text: "Hello" },
    })

    const chunks = translateAnthropicEventToOpenAIChunks(event, state)

    expect(chunks).toHaveLength(1)
    expect(chunks[0].choices[0].delta.content).toBe("Hello")
  })

  test("emits tool_call chunk on tool_use content_block_start", () => {
    const state = createOpenAIStreamState()
    state.model = "claude-sonnet-4-20250514"

    const event = makeEvent("content_block_start", {
      index: 1,
      content_block: {
        type: "tool_use",
        id: "toolu_123",
        name: "get_weather",
      },
    })

    const chunks = translateAnthropicEventToOpenAIChunks(event, state)

    expect(chunks).toHaveLength(1)
    const tc = chunks[0].choices[0].delta.tool_calls
    expect(tc).toBeDefined()
    expect(tc![0].id).toBe("toolu_123")
    expect(tc![0].function!.name).toBe("get_weather")
  })

  test("emits tool arguments on input_json_delta", () => {
    const state = createOpenAIStreamState()
    state.model = "claude-sonnet-4-20250514"

    // Start tool block
    translateAnthropicEventToOpenAIChunks(
      makeEvent("content_block_start", {
        index: 1,
        content_block: { type: "tool_use", id: "toolu_123", name: "get_weather" },
      }),
      state,
    )

    const event = makeEvent("content_block_delta", {
      index: 1,
      delta: { type: "input_json_delta", partial_json: '{"city":"Tok' },
    })

    const chunks = translateAnthropicEventToOpenAIChunks(event, state)

    expect(chunks).toHaveLength(1)
    const tc = chunks[0].choices[0].delta.tool_calls
    expect(tc).toBeDefined()
    expect(tc![0].function!.arguments).toBe('{"city":"Tok')
  })

  test("emits reasoning_text on thinking_delta", () => {
    const state = createOpenAIStreamState()
    state.model = "claude-sonnet-4-20250514"

    const event = makeEvent("content_block_delta", {
      index: 0,
      delta: { type: "thinking_delta", thinking: "Let me think..." },
    })

    const chunks = translateAnthropicEventToOpenAIChunks(event, state)

    expect(chunks).toHaveLength(1)
    expect(chunks[0].choices[0].delta.reasoning_text).toBe("Let me think...")
  })

  test("emits reasoning_opaque on signature_delta", () => {
    const state = createOpenAIStreamState()
    state.model = "claude-sonnet-4-20250514"

    const event = makeEvent("content_block_delta", {
      index: 0,
      delta: { type: "signature_delta", signature: "sig_abc" },
    })

    const chunks = translateAnthropicEventToOpenAIChunks(event, state)

    expect(chunks).toHaveLength(1)
    expect(chunks[0].choices[0].delta.reasoning_opaque).toBe("sig_abc")
  })

  test("emits finish_reason on message_delta", () => {
    const state = createOpenAIStreamState()
    state.model = "claude-sonnet-4-20250514"

    const event = makeEvent("message_delta", {
      delta: { stop_reason: "end_turn" },
      usage: { output_tokens: 50 },
    })

    const chunks = translateAnthropicEventToOpenAIChunks(event, state)

    expect(chunks).toHaveLength(1)
    expect(chunks[0].choices[0].finish_reason).toBe("stop")
  })

  test("emits nothing on message_stop", () => {
    const state = createOpenAIStreamState()

    const event = makeEvent("message_stop", {})
    const chunks = translateAnthropicEventToOpenAIChunks(event, state)

    expect(chunks).toHaveLength(0)
  })

  test("ignores content_block_start for text blocks", () => {
    const state = createOpenAIStreamState()
    state.model = "claude-sonnet-4-20250514"

    const event = makeEvent("content_block_start", {
      index: 0,
      content_block: { type: "text", text: "" },
    })

    const chunks = translateAnthropicEventToOpenAIChunks(event, state)
    expect(chunks).toHaveLength(0)
  })

  test("handles invalid JSON gracefully", () => {
    const state = createOpenAIStreamState()

    const chunks = translateAnthropicEventToOpenAIChunks(
      { event: "content_block_delta", data: "not json" },
      state,
    )
    expect(chunks).toHaveLength(0)
  })
})
