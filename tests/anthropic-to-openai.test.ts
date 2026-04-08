import { describe, expect, test } from "bun:test"
import { translateToOpenAI, translateToAnthropic } from "~/routes/messages/non-stream-translation"
import type { AnthropicMessagesPayload } from "~/lib/anthropic-types"
import type { ChatCompletionResponse } from "~/lib/openai-types"

describe("translateToOpenAI (Anthropic payload -> OpenAI payload)", () => {
  test("translates basic text message", () => {
    const payload: AnthropicMessagesPayload = {
      model: "gpt-4o",
      messages: [
        { role: "user", content: "Hello" },
      ],
      max_tokens: 1024,
    }

    const result = translateToOpenAI(payload)

    expect(result.model).toBe("gpt-4o")
    expect(result.max_tokens).toBe(1024)
    expect(result.messages).toHaveLength(1)
    expect(result.messages[0]).toEqual({ role: "user", content: "Hello" })
  })

  test("translates system prompt as string", () => {
    const payload: AnthropicMessagesPayload = {
      model: "gpt-4o",
      messages: [{ role: "user", content: "Hi" }],
      max_tokens: 1024,
      system: "You are helpful",
    }

    const result = translateToOpenAI(payload)

    expect(result.messages[0]).toEqual({ role: "system", content: "You are helpful" })
    expect(result.messages[1]).toEqual({ role: "user", content: "Hi" })
  })

  test("translates system prompt as TextBlock array", () => {
    const payload: AnthropicMessagesPayload = {
      model: "gpt-4o",
      messages: [{ role: "user", content: "Hi" }],
      max_tokens: 1024,
      system: [
        { type: "text", text: "Part one" },
        { type: "text", text: "Part two" },
      ],
    }

    const result = translateToOpenAI(payload)
    expect(result.messages[0]).toEqual({ role: "system", content: "Part one\n\nPart two" })
  })

  test("translates user message with tool_result blocks", () => {
    const payload: AnthropicMessagesPayload = {
      model: "gpt-4o",
      messages: [
        {
          role: "user",
          content: [
            { type: "tool_result", tool_use_id: "call_123", content: '{"result": 42}' },
            { type: "text", text: "What does that mean?" },
          ],
        },
      ],
      max_tokens: 1024,
    }

    const result = translateToOpenAI(payload)

    // tool_result comes first as role:tool, then remaining content as role:user
    expect(result.messages).toHaveLength(2)
    expect(result.messages[0].role).toBe("tool")
    expect(result.messages[0].tool_call_id).toBe("call_123")
    expect(result.messages[1].role).toBe("user")
  })

  test("translates assistant message with tool_use", () => {
    const payload: AnthropicMessagesPayload = {
      model: "gpt-4o",
      messages: [
        { role: "user", content: "Get weather" },
        {
          role: "assistant",
          content: [
            { type: "text", text: "Let me check." },
            {
              type: "tool_use",
              id: "call_456",
              name: "get_weather",
              input: { city: "Tokyo" },
            },
          ],
        },
      ],
      max_tokens: 1024,
    }

    const result = translateToOpenAI(payload)

    const assistantMsg = result.messages[1]
    expect(assistantMsg.role).toBe("assistant")
    expect(assistantMsg.tool_calls).toHaveLength(1)
    expect(assistantMsg.tool_calls![0].id).toBe("call_456")
    expect(assistantMsg.tool_calls![0].function.name).toBe("get_weather")
    expect(assistantMsg.tool_calls![0].function.arguments).toBe('{"city":"Tokyo"}')
  })

  test("translates assistant message with thinking blocks", () => {
    const payload: AnthropicMessagesPayload = {
      model: "gpt-4o",
      messages: [
        { role: "user", content: "Think about this" },
        {
          role: "assistant",
          content: [
            { type: "thinking", thinking: "Let me reason...", signature: "sig123" },
            { type: "text", text: "Here is my answer" },
          ],
        },
      ],
      max_tokens: 1024,
    }

    const result = translateToOpenAI(payload)

    const assistantMsg = result.messages[1]
    expect(assistantMsg.reasoning_text).toBe("Let me reason...")
    expect(assistantMsg.reasoning_opaque).toBe("sig123")
  })

  test("translates tools", () => {
    const payload: AnthropicMessagesPayload = {
      model: "gpt-4o",
      messages: [{ role: "user", content: "Hi" }],
      max_tokens: 1024,
      tools: [
        {
          name: "get_weather",
          description: "Get weather for a city",
          input_schema: {
            type: "object",
            properties: { city: { type: "string" } },
            required: ["city"],
          },
        },
      ],
    }

    const result = translateToOpenAI(payload)

    expect(result.tools).toHaveLength(1)
    expect(result.tools![0].type).toBe("function")
    expect(result.tools![0].function.name).toBe("get_weather")
    expect(result.tools![0].function.description).toBe("Get weather for a city")
  })

  test("translates tool_choice auto", () => {
    const payload: AnthropicMessagesPayload = {
      model: "gpt-4o",
      messages: [{ role: "user", content: "Hi" }],
      max_tokens: 1024,
      tool_choice: { type: "auto" },
    }

    const result = translateToOpenAI(payload)
    expect(result.tool_choice).toBe("auto")
  })

  test("translates tool_choice any -> required", () => {
    const payload: AnthropicMessagesPayload = {
      model: "gpt-4o",
      messages: [{ role: "user", content: "Hi" }],
      max_tokens: 1024,
      tool_choice: { type: "any" },
    }

    const result = translateToOpenAI(payload)
    expect(result.tool_choice).toBe("required")
  })

  test("translates tool_choice specific tool", () => {
    const payload: AnthropicMessagesPayload = {
      model: "gpt-4o",
      messages: [{ role: "user", content: "Hi" }],
      max_tokens: 1024,
      tool_choice: { type: "tool", name: "get_weather" },
    }

    const result = translateToOpenAI(payload)
    expect(result.tool_choice).toEqual({
      type: "function",
      function: { name: "get_weather" },
    })
  })

  test("translates tool_choice none", () => {
    const payload: AnthropicMessagesPayload = {
      model: "gpt-4o",
      messages: [{ role: "user", content: "Hi" }],
      max_tokens: 1024,
      tool_choice: { type: "none" },
    }

    const result = translateToOpenAI(payload)
    expect(result.tool_choice).toBe("none")
  })

  test("passes through thinking_budget", () => {
    const payload: AnthropicMessagesPayload = {
      model: "gpt-4o",
      messages: [{ role: "user", content: "Hi" }],
      max_tokens: 1024,
      thinking: { type: "enabled", budget_tokens: 8000 },
    }

    const result = translateToOpenAI(payload)
    expect(result.thinking_budget).toBe(8000)
  })

  test("passes through optional fields", () => {
    const payload: AnthropicMessagesPayload = {
      model: "gpt-4o",
      messages: [{ role: "user", content: "Hi" }],
      max_tokens: 1024,
      temperature: 0.7,
      top_p: 0.9,
      stop_sequences: ["END"],
      metadata: { user_id: "user_123" },
    }

    const result = translateToOpenAI(payload)
    expect(result.temperature).toBe(0.7)
    expect(result.top_p).toBe(0.9)
    expect(result.stop).toEqual(["END"])
    expect(result.user).toBe("user_123")
  })

  test("translates image content blocks", () => {
    const payload: AnthropicMessagesPayload = {
      model: "gpt-4o",
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "What is this?" },
            {
              type: "image",
              source: {
                type: "base64",
                media_type: "image/png",
                data: "iVBORw0KGgo=",
              },
            },
          ],
        },
      ],
      max_tokens: 1024,
    }

    const result = translateToOpenAI(payload)

    const content = result.messages[0].content as Array<Record<string, unknown>>
    expect(content).toHaveLength(2)
    expect(content[0]).toEqual({ type: "text", text: "What is this?" })
    expect(content[1]).toEqual({
      type: "image_url",
      image_url: { url: "data:image/png;base64,iVBORw0KGgo=" },
    })
  })
})

describe("translateToAnthropic (OpenAI response -> Anthropic response)", () => {
  test("translates basic text response", () => {
    const response: ChatCompletionResponse = {
      id: "chatcmpl-123",
      object: "chat.completion",
      created: 1700000000,
      model: "gpt-4o",
      choices: [
        {
          index: 0,
          message: { role: "assistant", content: "Hello!" },
          logprobs: null,
          finish_reason: "stop",
        },
      ],
      usage: {
        prompt_tokens: 10,
        completion_tokens: 5,
        total_tokens: 15,
      },
    }

    const result = translateToAnthropic(response)

    expect(result.id).toBe("chatcmpl-123")
    expect(result.type).toBe("message")
    expect(result.role).toBe("assistant")
    expect(result.model).toBe("gpt-4o")
    expect(result.stop_reason).toBe("end_turn")
    expect(result.content).toHaveLength(1)
    expect(result.content[0]).toEqual({ type: "text", text: "Hello!" })
    expect(result.usage.input_tokens).toBe(10)
    expect(result.usage.output_tokens).toBe(5)
  })

  test("translates response with tool_calls", () => {
    const response: ChatCompletionResponse = {
      id: "chatcmpl-456",
      object: "chat.completion",
      created: 1700000000,
      model: "gpt-4o",
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content: null,
            tool_calls: [
              {
                id: "call_abc",
                type: "function",
                function: {
                  name: "get_weather",
                  arguments: '{"city":"Tokyo"}',
                },
              },
            ],
          },
          logprobs: null,
          finish_reason: "tool_calls",
        },
      ],
    }

    const result = translateToAnthropic(response)

    expect(result.stop_reason).toBe("tool_use")
    const toolBlock = result.content.find((b) => b.type === "tool_use")
    expect(toolBlock).toBeDefined()
    if (toolBlock && toolBlock.type === "tool_use") {
      expect(toolBlock.id).toBe("call_abc")
      expect(toolBlock.name).toBe("get_weather")
      expect(toolBlock.input).toEqual({ city: "Tokyo" })
    }
  })

  test("translates response with reasoning", () => {
    const response: ChatCompletionResponse = {
      id: "chatcmpl-789",
      object: "chat.completion",
      created: 1700000000,
      model: "gpt-4o",
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content: "The answer is 42",
            reasoning_text: "Let me think step by step...",
            reasoning_opaque: "opaque_sig",
          },
          logprobs: null,
          finish_reason: "stop",
        },
      ],
    }

    const result = translateToAnthropic(response)

    const thinkBlock = result.content.find((b) => b.type === "thinking")
    expect(thinkBlock).toBeDefined()
    if (thinkBlock && thinkBlock.type === "thinking") {
      expect(thinkBlock.thinking).toBe("Let me think step by step...")
      expect(thinkBlock.signature).toBe("opaque_sig")
    }

    const textBlock = result.content.find((b) => b.type === "text")
    expect(textBlock).toBeDefined()
  })

  test("handles cached tokens in usage", () => {
    const response: ChatCompletionResponse = {
      id: "chatcmpl-cache",
      object: "chat.completion",
      created: 1700000000,
      model: "gpt-4o",
      choices: [
        {
          index: 0,
          message: { role: "assistant", content: "ok" },
          logprobs: null,
          finish_reason: "stop",
        },
      ],
      usage: {
        prompt_tokens: 100,
        completion_tokens: 10,
        total_tokens: 110,
        prompt_tokens_details: { cached_tokens: 80 },
      },
    }

    const result = translateToAnthropic(response)

    expect(result.usage.input_tokens).toBe(20) // 100 - 80
    expect(result.usage.cache_read_input_tokens).toBe(80)
  })
})
