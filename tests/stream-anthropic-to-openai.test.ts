import { describe, expect, test } from "bun:test"
import { translateChunkToAnthropicEvents } from "~/routes/messages/stream-translation"
import type { AnthropicStreamState } from "~/lib/anthropic-types"
import type { ChatCompletionChunk } from "~/lib/openai-types"

function createState(): AnthropicStreamState {
  return {
    messageStartSent: false,
    contentBlockIndex: 0,
    contentBlockOpen: false,
    thinkingBlockOpen: false,
    toolCalls: {},
  }
}

function makeChunk(overrides: Partial<ChatCompletionChunk> & { delta?: Record<string, unknown>; finish_reason?: string | null }): ChatCompletionChunk {
  return {
    id: "chatcmpl-test",
    object: "chat.completion.chunk",
    created: 1700000000,
    model: "gpt-4o",
    choices: [{
      index: 0,
      delta: (overrides.delta || {}) as ChatCompletionChunk["choices"][0]["delta"],
      finish_reason: (overrides.finish_reason ?? null) as ChatCompletionChunk["choices"][0]["finish_reason"],
      logprobs: null,
    }],
    ...("usage" in overrides ? { usage: overrides.usage } : {}),
  }
}

describe("translateChunkToAnthropicEvents (OpenAI stream -> Anthropic events)", () => {
  test("emits message_start on first chunk", () => {
    const state = createState()
    const chunk = makeChunk({ delta: { role: "assistant", content: "" } })

    const events = translateChunkToAnthropicEvents(chunk, state)

    const msgStart = events.find((e) => e.type === "message_start")
    expect(msgStart).toBeDefined()
    expect(state.messageStartSent).toBe(true)
  })

  test("does not emit message_start twice", () => {
    const state = createState()
    const chunk1 = makeChunk({ delta: { content: "Hello" } })
    const chunk2 = makeChunk({ delta: { content: " world" } })

    translateChunkToAnthropicEvents(chunk1, state)
    const events2 = translateChunkToAnthropicEvents(chunk2, state)

    const msgStarts = events2.filter((e) => e.type === "message_start")
    expect(msgStarts).toHaveLength(0)
  })

  test("emits text content block start + delta", () => {
    const state = createState()
    state.messageStartSent = true

    const chunk = makeChunk({ delta: { content: "Hello" } })
    const events = translateChunkToAnthropicEvents(chunk, state)

    const blockStart = events.find((e) => e.type === "content_block_start")
    expect(blockStart).toBeDefined()
    if (blockStart && blockStart.type === "content_block_start") {
      expect(blockStart.content_block).toEqual({ type: "text", text: "" })
    }

    const textDelta = events.find(
      (e) => e.type === "content_block_delta" && "delta" in e && (e.delta as Record<string, unknown>).type === "text_delta",
    )
    expect(textDelta).toBeDefined()
  })

  test("streams multiple text deltas without reopening block", () => {
    const state = createState()
    state.messageStartSent = true

    translateChunkToAnthropicEvents(makeChunk({ delta: { content: "Hello" } }), state)
    const events = translateChunkToAnthropicEvents(makeChunk({ delta: { content: " world" } }), state)

    const blockStarts = events.filter((e) => e.type === "content_block_start")
    expect(blockStarts).toHaveLength(0) // block already open
  })

  test("emits thinking block on reasoning_text", () => {
    const state = createState()
    state.messageStartSent = true

    const chunk = makeChunk({ delta: { reasoning_text: "Let me think..." } })
    const events = translateChunkToAnthropicEvents(chunk, state)

    const blockStart = events.find((e) => e.type === "content_block_start")
    expect(blockStart).toBeDefined()
    if (blockStart && blockStart.type === "content_block_start") {
      expect(blockStart.content_block).toEqual({ type: "thinking", thinking: "" })
    }
    expect(state.thinkingBlockOpen).toBe(true)
  })

  test("closes thinking block when content starts", () => {
    const state = createState()
    state.messageStartSent = true

    translateChunkToAnthropicEvents(makeChunk({ delta: { reasoning_text: "thinking..." } }), state)
    expect(state.thinkingBlockOpen).toBe(true)

    const events = translateChunkToAnthropicEvents(makeChunk({ delta: { content: "Answer" } }), state)
    expect(state.thinkingBlockOpen).toBe(false)

    const sigDelta = events.find(
      (e) => e.type === "content_block_delta" && "delta" in e && (e.delta as Record<string, unknown>).type === "signature_delta",
    )
    expect(sigDelta).toBeDefined()

    const blockStop = events.find((e) => e.type === "content_block_stop")
    expect(blockStop).toBeDefined()
  })

  test("emits tool_use block on tool_calls delta", () => {
    const state = createState()
    state.messageStartSent = true

    const chunk = makeChunk({
      delta: {
        tool_calls: [{
          index: 0,
          id: "call_abc",
          type: "function",
          function: { name: "get_weather", arguments: "" },
        }],
      },
    })

    const events = translateChunkToAnthropicEvents(chunk, state)

    const blockStart = events.find((e) => e.type === "content_block_start")
    expect(blockStart).toBeDefined()
    if (blockStart && blockStart.type === "content_block_start") {
      expect(blockStart.content_block).toEqual({
        type: "tool_use",
        id: "call_abc",
        name: "get_weather",
        input: {},
      })
    }
  })

  test("emits input_json_delta for tool arguments", () => {
    const state = createState()
    state.messageStartSent = true

    // First chunk: tool start
    translateChunkToAnthropicEvents(makeChunk({
      delta: {
        tool_calls: [{
          index: 0,
          id: "call_abc",
          type: "function",
          function: { name: "get_weather", arguments: "" },
        }],
      },
    }), state)

    // Second chunk: arguments
    const events = translateChunkToAnthropicEvents(makeChunk({
      delta: {
        tool_calls: [{
          index: 0,
          function: { arguments: '{"city":"Tok' },
        }],
      },
    }), state)

    const jsonDelta = events.find(
      (e) => e.type === "content_block_delta" && "delta" in e && (e.delta as Record<string, unknown>).type === "input_json_delta",
    )
    expect(jsonDelta).toBeDefined()
    if (jsonDelta && jsonDelta.type === "content_block_delta" && (jsonDelta.delta as Record<string, unknown>).type === "input_json_delta") {
      expect((jsonDelta.delta as Record<string, unknown>).partial_json).toBe('{"city":"Tok')
    }
  })

  test("emits message_delta + message_stop on finish", () => {
    const state = createState()
    state.messageStartSent = true

    // Open a text block first
    translateChunkToAnthropicEvents(makeChunk({ delta: { content: "Hi" } }), state)

    const events = translateChunkToAnthropicEvents(makeChunk({
      delta: {},
      finish_reason: "stop",
      usage: {
        prompt_tokens: 10,
        completion_tokens: 5,
        total_tokens: 15,
      },
    }), state)

    const msgDelta = events.find((e) => e.type === "message_delta")
    expect(msgDelta).toBeDefined()
    if (msgDelta && msgDelta.type === "message_delta") {
      expect(msgDelta.delta.stop_reason).toBe("end_turn")
    }

    const msgStop = events.find((e) => e.type === "message_stop")
    expect(msgStop).toBeDefined()
  })

  test("handles empty choices array", () => {
    const state = createState()
    const chunk: ChatCompletionChunk = {
      id: "chatcmpl-test",
      object: "chat.completion.chunk",
      created: 1700000000,
      model: "gpt-4o",
      choices: [],
    }

    const events = translateChunkToAnthropicEvents(chunk, state)
    expect(events).toHaveLength(0)
  })

  test("handles multiple tool calls in sequence", () => {
    const state = createState()
    state.messageStartSent = true

    // First tool
    translateChunkToAnthropicEvents(makeChunk({
      delta: {
        tool_calls: [{
          index: 0,
          id: "call_1",
          type: "function",
          function: { name: "tool_a", arguments: '{}' },
        }],
      },
    }), state)

    // Second tool
    const events = translateChunkToAnthropicEvents(makeChunk({
      delta: {
        tool_calls: [{
          index: 1,
          id: "call_2",
          type: "function",
          function: { name: "tool_b", arguments: '{}' },
        }],
      },
    }), state)

    // Should close first tool block and open second
    const blockStop = events.find((e) => e.type === "content_block_stop")
    expect(blockStop).toBeDefined()

    const blockStart = events.find((e) => e.type === "content_block_start")
    expect(blockStart).toBeDefined()

    expect(Object.keys(state.toolCalls)).toHaveLength(2)
  })
})
