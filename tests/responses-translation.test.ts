import { describe, expect, test } from "bun:test"
import {
  translateAnthropicToResponsesPayload,
  translateResponsesToAnthropic,
} from "~/routes/responses/translation"
import type { AnthropicMessagesPayload } from "~/lib/anthropic-types"
import type { ResponsesResult } from "~/lib/responses-types"

describe("translateAnthropicToResponsesPayload", () => {
  test("translates basic text message", () => {
    const payload: AnthropicMessagesPayload = {
      model: "o3",
      messages: [{ role: "user", content: "Hello" }],
      max_tokens: 4096,
    }

    const result = translateAnthropicToResponsesPayload(payload)

    expect(result.model).toBe("o3")
    expect(result.max_output_tokens).toBe(4096)
    expect(result.input).toHaveLength(1)
    const msg = result.input![0] as Record<string, unknown>
    expect(msg.role).toBe("user")
    expect(msg.content).toBe("Hello")
  })

  test("translates system prompt to instructions", () => {
    const payload: AnthropicMessagesPayload = {
      model: "o3",
      messages: [{ role: "user", content: "Hi" }],
      max_tokens: 4096,
      system: "You are helpful",
    }

    const result = translateAnthropicToResponsesPayload(payload)
    expect(result.instructions).toBe("You are helpful")
  })

  test("translates system array to instructions string", () => {
    const payload: AnthropicMessagesPayload = {
      model: "o3",
      messages: [{ role: "user", content: "Hi" }],
      max_tokens: 4096,
      system: [{ type: "text", text: "Part 1" }, { type: "text", text: "Part 2" }],
    }

    const result = translateAnthropicToResponsesPayload(payload)
    expect(result.instructions).toBe("Part 1 Part 2")
  })

  test("translates user message with content blocks", () => {
    const payload: AnthropicMessagesPayload = {
      model: "o3",
      messages: [{
        role: "user",
        content: [
          { type: "text", text: "What is this?" },
          { type: "image", source: { type: "base64", media_type: "image/png", data: "abc123" } },
        ],
      }],
      max_tokens: 4096,
    }

    const result = translateAnthropicToResponsesPayload(payload)

    const msg = result.input![0] as Record<string, unknown>
    const content = msg.content as Array<Record<string, unknown>>
    expect(content).toHaveLength(2)
    expect(content[0]).toEqual({ type: "input_text", text: "What is this?" })
    expect(content[1]).toEqual({ type: "input_image", image_url: "data:image/png;base64,abc123", detail: "auto" })
  })

  test("translates tool_result to function_call_output", () => {
    const payload: AnthropicMessagesPayload = {
      model: "o3",
      messages: [{
        role: "user",
        content: [
          { type: "tool_result", tool_use_id: "call_123", content: '{"temp": 20}' },
          { type: "text", text: "What does that mean?" },
        ],
      }],
      max_tokens: 4096,
    }

    const result = translateAnthropicToResponsesPayload(payload)

    const items = result.input!
    const funcOutput = items.find((i) => (i as Record<string, unknown>).type === "function_call_output")
    expect(funcOutput).toBeDefined()
    expect((funcOutput as Record<string, unknown>).call_id).toBe("call_123")
  })

  test("translates tool_use to function_call", () => {
    const payload: AnthropicMessagesPayload = {
      model: "o3",
      messages: [{
        role: "assistant",
        content: [
          { type: "tool_use", id: "call_456", name: "get_weather", input: { city: "Tokyo" } },
        ],
      }],
      max_tokens: 4096,
    }

    const result = translateAnthropicToResponsesPayload(payload)

    const funcCall = result.input!.find((i) => (i as Record<string, unknown>).type === "function_call")
    expect(funcCall).toBeDefined()
    expect((funcCall as Record<string, unknown>).name).toBe("get_weather")
    expect((funcCall as Record<string, unknown>).arguments).toBe('{"city":"Tokyo"}')
  })

  test("translates thinking with @ signature to reasoning", () => {
    const payload: AnthropicMessagesPayload = {
      model: "o3",
      messages: [{
        role: "assistant",
        content: [
          { type: "thinking", thinking: "Let me think...", signature: "encrypted_data@reasoning_id" },
          { type: "text", text: "Answer" },
        ],
      }],
      max_tokens: 4096,
    }

    const result = translateAnthropicToResponsesPayload(payload)

    const reasoning = result.input!.find((i) => (i as Record<string, unknown>).type === "reasoning")
    expect(reasoning).toBeDefined()
    expect((reasoning as Record<string, unknown>).encrypted_content).toBe("encrypted_data")
    expect((reasoning as Record<string, unknown>).id).toBe("reasoning_id")
  })

  test("translates tools", () => {
    const payload: AnthropicMessagesPayload = {
      model: "o3",
      messages: [{ role: "user", content: "Hi" }],
      max_tokens: 4096,
      tools: [{
        name: "get_weather",
        description: "Get weather",
        input_schema: { type: "object", properties: { city: { type: "string" } } },
      }],
    }

    const result = translateAnthropicToResponsesPayload(payload)

    expect(result.tools).toHaveLength(1)
    expect(result.tools![0].type).toBe("function")
    expect(result.tools![0].name).toBe("get_weather")
    expect(result.tools![0].strict).toBe(false)
  })

  test("translates tool_choice", () => {
    const payload: AnthropicMessagesPayload = {
      model: "o3",
      messages: [{ role: "user", content: "Hi" }],
      max_tokens: 4096,
      tool_choice: { type: "any" },
    }

    const result = translateAnthropicToResponsesPayload(payload)
    expect(result.tool_choice).toBe("required")
  })

  test("enables reasoning when thinking is set", () => {
    const payload: AnthropicMessagesPayload = {
      model: "o3",
      messages: [{ role: "user", content: "Hi" }],
      max_tokens: 4096,
      thinking: { type: "enabled", budget_tokens: 5000 },
    }

    const result = translateAnthropicToResponsesPayload(payload)
    expect(result.reasoning).toEqual({ effort: "high", summary: "detailed" })
  })
})

describe("translateResponsesToAnthropic", () => {
  test("translates basic text response", () => {
    const response: ResponsesResult = {
      id: "resp_123",
      object: "response",
      created_at: 1700000000,
      model: "o3",
      output: [{
        id: "msg_1",
        type: "message",
        role: "assistant",
        status: "completed",
        content: [{ type: "output_text", text: "Hello!", annotations: [] }],
      }],
      output_text: "Hello!",
      status: "completed",
      usage: { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
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

    const result = translateResponsesToAnthropic(response)

    expect(result.id).toBe("resp_123")
    expect(result.model).toBe("o3")
    expect(result.stop_reason).toBe("end_turn")
    expect(result.content).toHaveLength(1)
    expect(result.content[0]).toEqual({ type: "text", text: "Hello!" })
    expect(result.usage.input_tokens).toBe(10)
    expect(result.usage.output_tokens).toBe(5)
  })

  test("translates function_call response", () => {
    const response: ResponsesResult = {
      id: "resp_456",
      object: "response",
      created_at: 1700000000,
      model: "o3",
      output: [{
        id: "fc_1",
        type: "function_call",
        call_id: "call_abc",
        name: "get_weather",
        arguments: '{"city":"Tokyo"}',
        status: "completed",
      }],
      output_text: "",
      status: "completed",
      usage: { input_tokens: 20, output_tokens: 15, total_tokens: 35 },
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

    const result = translateResponsesToAnthropic(response)

    expect(result.stop_reason).toBe("tool_use")
    const toolBlock = result.content.find((b) => b.type === "tool_use")
    expect(toolBlock).toBeDefined()
    if (toolBlock && toolBlock.type === "tool_use") {
      expect(toolBlock.id).toBe("call_abc")
      expect(toolBlock.name).toBe("get_weather")
      expect(toolBlock.input).toEqual({ city: "Tokyo" })
    }
  })

  test("translates reasoning output to thinking block", () => {
    const response: ResponsesResult = {
      id: "resp_789",
      object: "response",
      created_at: 1700000000,
      model: "o3",
      output: [
        {
          id: "reasoning_1",
          type: "reasoning",
          summary: [{ type: "summary_text", text: "Step by step..." }],
          encrypted_content: "enc_data",
          status: "completed",
        },
        {
          id: "msg_1",
          type: "message",
          role: "assistant",
          status: "completed",
          content: [{ type: "output_text", text: "The answer is 42", annotations: [] }],
        },
      ],
      output_text: "The answer is 42",
      status: "completed",
      usage: { input_tokens: 10, output_tokens: 50, total_tokens: 60 },
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

    const result = translateResponsesToAnthropic(response)

    expect(result.content).toHaveLength(2)
    const thinkBlock = result.content.find((b) => b.type === "thinking")
    expect(thinkBlock).toBeDefined()
    if (thinkBlock && thinkBlock.type === "thinking") {
      expect(thinkBlock.thinking).toBe("Step by step...")
      expect(thinkBlock.signature).toBe("enc_data@reasoning_1")
    }
  })

  test("maps incomplete status with max_output_tokens", () => {
    const response: ResponsesResult = {
      id: "resp_inc",
      object: "response",
      created_at: 1700000000,
      model: "o3",
      output: [{
        id: "msg_1",
        type: "message",
        role: "assistant",
        status: "incomplete",
        content: [{ type: "output_text", text: "Partial", annotations: [] }],
      }],
      output_text: "Partial",
      status: "incomplete",
      usage: { input_tokens: 10, output_tokens: 100, total_tokens: 110 },
      error: null,
      incomplete_details: { reason: "max_output_tokens" },
      instructions: null,
      metadata: null,
      parallel_tool_calls: true,
      temperature: null,
      tool_choice: "auto",
      tools: [],
      top_p: null,
    }

    const result = translateResponsesToAnthropic(response)
    expect(result.stop_reason).toBe("max_tokens")
  })

  test("handles cached tokens in usage", () => {
    const response: ResponsesResult = {
      id: "resp_cache",
      object: "response",
      created_at: 1700000000,
      model: "o3",
      output: [{
        id: "msg_1",
        type: "message",
        role: "assistant",
        status: "completed",
        content: [{ type: "output_text", text: "ok", annotations: [] }],
      }],
      output_text: "ok",
      status: "completed",
      usage: {
        input_tokens: 100,
        output_tokens: 5,
        total_tokens: 105,
        input_tokens_details: { cached_tokens: 80 },
      },
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

    const result = translateResponsesToAnthropic(response)
    expect(result.usage.input_tokens).toBe(20) // 100 - 80
    expect(result.usage.cache_read_input_tokens).toBe(80)
  })

  test("falls back to output_text when no output items", () => {
    const response: ResponsesResult = {
      id: "resp_fb",
      object: "response",
      created_at: 1700000000,
      model: "o3",
      output: [],
      output_text: "Fallback text",
      status: "completed",
      usage: { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
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

    const result = translateResponsesToAnthropic(response)
    expect(result.content).toHaveLength(1)
    expect(result.content[0]).toEqual({ type: "text", text: "Fallback text" })
  })
})
