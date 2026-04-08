import { describe, expect, test } from "bun:test"
import {
  translateResponsesStreamEvent,
  createResponsesStreamState,
} from "~/routes/responses/stream-translation"
import type { ResponseStreamEvent, ResponsesResult } from "~/lib/responses-types"

const baseResponse: ResponsesResult = {
  id: "resp_test",
  object: "response",
  created_at: 1700000000,
  model: "o3",
  output: [],
  output_text: "",
  status: "completed",
  usage: { input_tokens: 10, output_tokens: 0, total_tokens: 10 },
  error: null,
  incomplete_details: null,
  instructions: null,
  metadata: null,
  parallel_tool_calls: true,
  temperature: null,
  tool_choice: "auto",
  tools: [],
  top_p: null,
}

describe("translateResponsesStreamEvent", () => {
  test("emits message_start on response.created", () => {
    const state = createResponsesStreamState()
    const event: ResponseStreamEvent = {
      type: "response.created",
      response: { ...baseResponse, model: "o3" },
      sequence_number: 0,
    }

    const events = translateResponsesStreamEvent(event, state)

    const msgStart = events.find((e) => e.type === "message_start")
    expect(msgStart).toBeDefined()
    expect(state.messageStartSent).toBe(true)
  })

  test("emits text_delta on response.output_text.delta", () => {
    const state = createResponsesStreamState()
    state.messageStartSent = true

    const event: ResponseStreamEvent = {
      type: "response.output_text.delta",
      delta: "Hello",
      content_index: 0,
      item_id: "msg_1",
      output_index: 0,
      sequence_number: 1,
    }

    const events = translateResponsesStreamEvent(event, state)

    const blockStart = events.find((e) => e.type === "content_block_start")
    expect(blockStart).toBeDefined()

    const textDelta = events.find(
      (e) => e.type === "content_block_delta" && "delta" in e && (e.delta as Record<string, unknown>).type === "text_delta",
    )
    expect(textDelta).toBeDefined()
  })

  test("emits thinking_delta on response.reasoning_summary_text.delta", () => {
    const state = createResponsesStreamState()
    state.messageStartSent = true

    const event: ResponseStreamEvent = {
      type: "response.reasoning_summary_text.delta",
      delta: "Thinking...",
      item_id: "reasoning_1",
      output_index: 0,
      sequence_number: 1,
      summary_index: 0,
    }

    const events = translateResponsesStreamEvent(event, state)

    const blockStart = events.find((e) => e.type === "content_block_start")
    expect(blockStart).toBeDefined()
    if (blockStart && blockStart.type === "content_block_start") {
      expect(blockStart.content_block).toEqual({ type: "thinking", thinking: "" })
    }

    const thinkDelta = events.find(
      (e) => e.type === "content_block_delta" && "delta" in e && (e.delta as Record<string, unknown>).type === "thinking_delta",
    )
    expect(thinkDelta).toBeDefined()
  })

  test("emits tool_use block on response.output_item.added for function_call", () => {
    const state = createResponsesStreamState()
    state.messageStartSent = true

    const event: ResponseStreamEvent = {
      type: "response.output_item.added",
      item: {
        id: "fc_1",
        type: "function_call",
        call_id: "call_abc",
        name: "get_weather",
        arguments: "",
        status: "in_progress",
      },
      output_index: 0,
      sequence_number: 1,
    }

    const events = translateResponsesStreamEvent(event, state)

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

  test("emits input_json_delta on function_call_arguments.delta", () => {
    const state = createResponsesStreamState()
    state.messageStartSent = true

    // First add the tool
    translateResponsesStreamEvent({
      type: "response.output_item.added",
      item: {
        id: "fc_1",
        type: "function_call",
        call_id: "call_abc",
        name: "get_weather",
        arguments: "",
        status: "in_progress",
      },
      output_index: 0,
      sequence_number: 1,
    }, state)

    const event: ResponseStreamEvent = {
      type: "response.function_call_arguments.delta",
      delta: '{"city":"Tok',
      item_id: "fc_1",
      output_index: 0,
      sequence_number: 2,
    }

    const events = translateResponsesStreamEvent(event, state)

    const jsonDelta = events.find(
      (e) => e.type === "content_block_delta" && "delta" in e && (e.delta as Record<string, unknown>).type === "input_json_delta",
    )
    expect(jsonDelta).toBeDefined()
  })

  test("emits signature_delta on output_item.done for reasoning", () => {
    const state = createResponsesStreamState()
    state.messageStartSent = true

    const event: ResponseStreamEvent = {
      type: "response.output_item.done",
      item: {
        id: "reasoning_1",
        type: "reasoning",
        summary: [{ type: "summary_text", text: "thought" }],
        encrypted_content: "enc_data",
        status: "completed",
      },
      output_index: 0,
      sequence_number: 3,
    }

    const events = translateResponsesStreamEvent(event, state)

    const sigDelta = events.find(
      (e) => e.type === "content_block_delta" && "delta" in e && (e.delta as Record<string, unknown>).type === "signature_delta",
    )
    expect(sigDelta).toBeDefined()
    if (sigDelta && sigDelta.type === "content_block_delta") {
      expect((sigDelta.delta as Record<string, unknown>).signature).toBe("enc_data@reasoning_1")
    }
  })

  test("emits message_delta + message_stop on response.completed", () => {
    const state = createResponsesStreamState()
    state.messageStartSent = true

    const event: ResponseStreamEvent = {
      type: "response.completed",
      response: {
        ...baseResponse,
        output: [{
          id: "msg_1",
          type: "message",
          role: "assistant",
          status: "completed",
          content: [{ type: "output_text", text: "Done", annotations: [] }],
        }],
        output_text: "Done",
        usage: { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
      },
      sequence_number: 10,
    }

    const events = translateResponsesStreamEvent(event, state)

    const msgDelta = events.find((e) => e.type === "message_delta")
    expect(msgDelta).toBeDefined()
    if (msgDelta && msgDelta.type === "message_delta") {
      expect(msgDelta.delta.stop_reason).toBe("end_turn")
    }

    const msgStop = events.find((e) => e.type === "message_stop")
    expect(msgStop).toBeDefined()
    expect(state.messageCompleted).toBe(true)
  })

  test("emits error on response.failed", () => {
    const state = createResponsesStreamState()
    state.messageStartSent = true

    const event: ResponseStreamEvent = {
      type: "response.failed",
      response: {
        ...baseResponse,
        status: "failed",
        error: { message: "Something went wrong" },
      },
      sequence_number: 5,
    }

    const events = translateResponsesStreamEvent(event, state)

    const errorEvent = events.find((e) => e.type === "error")
    expect(errorEvent).toBeDefined()
    expect(state.messageCompleted).toBe(true)
  })

  test("emits error on error event", () => {
    const state = createResponsesStreamState()

    const event: ResponseStreamEvent = {
      type: "error",
      code: "server_error",
      message: "Internal error",
      param: null,
      sequence_number: 1,
    }

    const events = translateResponsesStreamEvent(event, state)

    const errorEvent = events.find((e) => e.type === "error")
    expect(errorEvent).toBeDefined()
    if (errorEvent && errorEvent.type === "error") {
      expect(errorEvent.error.message).toBe("Internal error")
    }
  })

  test("ignores non-function_call output_item.added", () => {
    const state = createResponsesStreamState()
    state.messageStartSent = true

    const event: ResponseStreamEvent = {
      type: "response.output_item.added",
      item: {
        id: "msg_1",
        type: "message",
        role: "assistant",
        status: "in_progress",
      },
      output_index: 0,
      sequence_number: 1,
    }

    const events = translateResponsesStreamEvent(event, state)
    expect(events).toHaveLength(0)
  })
})
