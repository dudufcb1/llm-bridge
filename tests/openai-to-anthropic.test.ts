import { describe, expect, test } from "bun:test"
import {
  translateToAnthropic,
  translateToOpenAI,
} from "~/routes/chat-completions/openai-to-anthropic"
import type { ChatCompletionsPayload, ChatCompletionResponse } from "~/lib/openai-types"
import type { AnthropicResponse } from "~/lib/anthropic-types"

describe("translateToAnthropic (OpenAI payload -> Anthropic payload)", () => {
  test("translates basic text message", () => {
    const payload: ChatCompletionsPayload = {
      model: "claude-sonnet-4-20250514",
      messages: [
        { role: "user", content: "Hello" },
      ],
      max_tokens: 1024,
    }

    const result = translateToAnthropic(payload)

    expect(result.model).toBe("claude-sonnet-4-20250514")
    expect(result.max_tokens).toBe(1024)
    expect(result.messages).toHaveLength(1)
    expect(result.messages[0].role).toBe("user")
  })

  test("extracts system message", () => {
    const payload: ChatCompletionsPayload = {
      model: "claude-sonnet-4-20250514",
      messages: [
        { role: "system", content: "You are helpful" },
        { role: "user", content: "Hi" },
      ],
      max_tokens: 1024,
    }

    const result = translateToAnthropic(payload)

    expect(result.system).toBe("You are helpful")
    expect(result.messages).toHaveLength(1)
    expect(result.messages[0].role).toBe("user")
  })

  test("merges multiple system messages", () => {
    const payload: ChatCompletionsPayload = {
      model: "claude-sonnet-4-20250514",
      messages: [
        { role: "system", content: "Rule 1" },
        { role: "developer", content: "Rule 2" },
        { role: "user", content: "Hi" },
      ],
      max_tokens: 1024,
    }

    const result = translateToAnthropic(payload)

    expect(result.system).toBe("Rule 1\n\nRule 2")
  })

  test("translates tool messages as tool_result", () => {
    const payload: ChatCompletionsPayload = {
      model: "claude-sonnet-4-20250514",
      messages: [
        { role: "user", content: "Get weather" },
        {
          role: "assistant",
          content: null,
          tool_calls: [{
            id: "call_123",
            type: "function",
            function: { name: "get_weather", arguments: '{"city":"Tokyo"}' },
          }],
        },
        {
          role: "tool",
          content: '{"temp": 20}',
          tool_call_id: "call_123",
        },
        { role: "user", content: "What is that?" },
      ],
      max_tokens: 1024,
    }

    const result = translateToAnthropic(payload)

    // assistant with tool_use
    expect(result.messages[1].role).toBe("assistant")
    const assistantContent = result.messages[1].content as Array<Record<string, unknown>>
    const toolUse = assistantContent.find((b) => b.type === "tool_use")
    expect(toolUse).toBeDefined()

    // tool result merged into user message
    expect(result.messages[2].role).toBe("user")
    const userContent = result.messages[2].content as Array<Record<string, unknown>>
    const toolResult = userContent.find((b) => b.type === "tool_result")
    expect(toolResult).toBeDefined()
  })

  test("translates image_url to base64 image block", () => {
    const payload: ChatCompletionsPayload = {
      model: "claude-sonnet-4-20250514",
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "What is this?" },
            {
              type: "image_url",
              image_url: { url: "data:image/png;base64,iVBORw0KGgo=" },
            },
          ],
        },
      ],
      max_tokens: 1024,
    }

    const result = translateToAnthropic(payload)

    const userContent = result.messages[0].content as Array<Record<string, unknown>>
    expect(userContent).toHaveLength(2)
    expect(userContent[0]).toEqual({ type: "text", text: "What is this?" })
    expect(userContent[1]).toEqual({
      type: "image",
      source: {
        type: "base64",
        media_type: "image/png",
        data: "iVBORw0KGgo=",
      },
    })
  })

  test("translates tools", () => {
    const payload: ChatCompletionsPayload = {
      model: "claude-sonnet-4-20250514",
      messages: [{ role: "user", content: "Hi" }],
      max_tokens: 1024,
      tools: [{
        type: "function",
        function: {
          name: "get_weather",
          description: "Get weather",
          parameters: { type: "object", properties: { city: { type: "string" } } },
        },
      }],
    }

    const result = translateToAnthropic(payload)

    expect(result.tools).toHaveLength(1)
    expect(result.tools![0].name).toBe("get_weather")
    expect(result.tools![0].description).toBe("Get weather")
  })

  test("translates tool_choice auto", () => {
    const payload: ChatCompletionsPayload = {
      model: "claude-sonnet-4-20250514",
      messages: [{ role: "user", content: "Hi" }],
      max_tokens: 1024,
      tool_choice: "auto",
    }

    const result = translateToAnthropic(payload)
    expect(result.tool_choice).toEqual({ type: "auto" })
  })

  test("translates tool_choice required -> any", () => {
    const payload: ChatCompletionsPayload = {
      model: "claude-sonnet-4-20250514",
      messages: [{ role: "user", content: "Hi" }],
      max_tokens: 1024,
      tool_choice: "required",
    }

    const result = translateToAnthropic(payload)
    expect(result.tool_choice).toEqual({ type: "any" })
  })

  test("translates tool_choice specific function", () => {
    const payload: ChatCompletionsPayload = {
      model: "claude-sonnet-4-20250514",
      messages: [{ role: "user", content: "Hi" }],
      max_tokens: 1024,
      tool_choice: { type: "function", function: { name: "get_weather" } },
    }

    const result = translateToAnthropic(payload)
    expect(result.tool_choice).toEqual({ type: "tool", name: "get_weather" })
  })

  test("translates thinking_budget", () => {
    const payload: ChatCompletionsPayload = {
      model: "claude-sonnet-4-20250514",
      messages: [{ role: "user", content: "Hi" }],
      max_tokens: 1024,
      thinking_budget: 5000,
    }

    const result = translateToAnthropic(payload)
    expect(result.thinking).toEqual({ type: "enabled", budget_tokens: 5000 })
  })

  test("passes through optional fields", () => {
    const payload: ChatCompletionsPayload = {
      model: "claude-sonnet-4-20250514",
      messages: [{ role: "user", content: "Hi" }],
      max_tokens: 1024,
      temperature: 0.5,
      top_p: 0.8,
      stop: ["END", "STOP"],
      user: "user_456",
    }

    const result = translateToAnthropic(payload)
    expect(result.temperature).toBe(0.5)
    expect(result.top_p).toBe(0.8)
    expect(result.stop_sequences).toEqual(["END", "STOP"])
    expect(result.metadata).toEqual({ user_id: "user_456" })
  })

  test("handles single stop string", () => {
    const payload: ChatCompletionsPayload = {
      model: "claude-sonnet-4-20250514",
      messages: [{ role: "user", content: "Hi" }],
      max_tokens: 1024,
      stop: "END",
    }

    const result = translateToAnthropic(payload)
    expect(result.stop_sequences).toEqual(["END"])
  })

  test("defaults max_tokens to 4096 when null", () => {
    const payload: ChatCompletionsPayload = {
      model: "claude-sonnet-4-20250514",
      messages: [{ role: "user", content: "Hi" }],
    }

    const result = translateToAnthropic(payload)
    expect(result.max_tokens).toBe(4096)
  })
})

describe("translateToOpenAI (Anthropic response -> OpenAI response)", () => {
  test("translates basic text response", () => {
    const response: AnthropicResponse = {
      id: "msg_123",
      type: "message",
      role: "assistant",
      model: "claude-sonnet-4-20250514",
      content: [{ type: "text", text: "Hello!" }],
      stop_reason: "end_turn",
      stop_sequence: null,
      usage: { input_tokens: 10, output_tokens: 5 },
    }

    const result = translateToOpenAI(response)

    expect(result.id).toBe("msg_123")
    expect(result.object).toBe("chat.completion")
    expect(result.model).toBe("claude-sonnet-4-20250514")
    expect(result.choices).toHaveLength(1)
    expect(result.choices[0].message.content).toBe("Hello!")
    expect(result.choices[0].finish_reason).toBe("stop")
    expect(result.usage?.prompt_tokens).toBe(10)
    expect(result.usage?.completion_tokens).toBe(5)
    expect(result.usage?.total_tokens).toBe(15)
  })

  test("translates tool_use response", () => {
    const response: AnthropicResponse = {
      id: "msg_456",
      type: "message",
      role: "assistant",
      model: "claude-sonnet-4-20250514",
      content: [
        {
          type: "tool_use",
          id: "toolu_123",
          name: "get_weather",
          input: { city: "Tokyo" },
        },
      ],
      stop_reason: "tool_use",
      stop_sequence: null,
      usage: { input_tokens: 20, output_tokens: 15 },
    }

    const result = translateToOpenAI(response)

    expect(result.choices[0].finish_reason).toBe("tool_calls")
    expect(result.choices[0].message.tool_calls).toHaveLength(1)
    expect(result.choices[0].message.tool_calls![0].id).toBe("toolu_123")
    expect(result.choices[0].message.tool_calls![0].function.name).toBe("get_weather")
    expect(result.choices[0].message.tool_calls![0].function.arguments).toBe('{"city":"Tokyo"}')
  })

  test("translates thinking blocks to reasoning fields", () => {
    const response: AnthropicResponse = {
      id: "msg_789",
      type: "message",
      role: "assistant",
      model: "claude-sonnet-4-20250514",
      content: [
        { type: "thinking", thinking: "Step by step...", signature: "sig_abc" },
        { type: "text", text: "The answer" },
      ],
      stop_reason: "end_turn",
      stop_sequence: null,
      usage: { input_tokens: 10, output_tokens: 50 },
    }

    const result = translateToOpenAI(response)

    expect(result.choices[0].message.reasoning_text).toBe("Step by step...")
    expect(result.choices[0].message.reasoning_opaque).toBe("sig_abc")
    expect(result.choices[0].message.content).toBe("The answer")
  })

  test("handles cached tokens", () => {
    const response: AnthropicResponse = {
      id: "msg_cache",
      type: "message",
      role: "assistant",
      model: "claude-sonnet-4-20250514",
      content: [{ type: "text", text: "ok" }],
      stop_reason: "end_turn",
      stop_sequence: null,
      usage: {
        input_tokens: 20,
        output_tokens: 5,
        cache_read_input_tokens: 80,
      },
    }

    const result = translateToOpenAI(response)

    expect(result.usage?.prompt_tokens).toBe(20)
    expect(result.usage?.prompt_tokens_details?.cached_tokens).toBe(80)
  })
})
