// Generic API clients for real OpenAI and Anthropic APIs

import consola from "consola"
import { events } from "fetch-event-stream"

import type { ChatCompletionsPayload, ChatCompletionResponse } from "./openai-types"
import type { AnthropicMessagesPayload, AnthropicResponse } from "./anthropic-types"
import type { ResponsesPayload, ResponsesResult } from "./responses-types"

export async function callOpenAI(
  payload: ChatCompletionsPayload,
  baseUrl: string,
  apiKey: string,
) {
  const url = `${baseUrl}/v1/chat/completions`
  consola.debug(`-> OpenAI: ${url} model=${payload.model}`)

  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  })

  if (!response.ok) {
    const body = await response.text()
    consola.error(`OpenAI error ${response.status}: ${body}`)
    throw new Error(`OpenAI API error: ${response.status} ${body}`)
  }

  if (payload.stream) {
    return events(response)
  }

  return (await response.json()) as ChatCompletionResponse
}

export async function callAnthropic(
  payload: AnthropicMessagesPayload,
  baseUrl: string,
  apiKey: string,
) {
  const url = `${baseUrl}/v1/messages`
  consola.debug(`-> Anthropic: ${url} model=${payload.model}`)

  const response = await fetch(url, {
    method: "POST",
    headers: {
      "x-api-key": apiKey,
      "anthropic-version": "2023-06-01",
      "content-type": "application/json",
    },
    body: JSON.stringify(payload),
  })

  if (!response.ok) {
    const body = await response.text()
    consola.error(`Anthropic error ${response.status}: ${body}`)
    throw new Error(`Anthropic API error: ${response.status} ${body}`)
  }

  if (payload.stream) {
    return events(response)
  }

  return (await response.json()) as AnthropicResponse
}

export async function callOpenAIResponses(
  payload: ResponsesPayload,
  baseUrl: string,
  apiKey: string,
) {
  const url = `${baseUrl}/v1/responses`
  consola.debug(`-> OpenAI Responses: ${url} model=${payload.model}`)

  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  })

  if (!response.ok) {
    const body = await response.text()
    consola.error(`OpenAI Responses error ${response.status}: ${body}`)
    throw new Error(`OpenAI Responses API error: ${response.status} ${body}`)
  }

  if (payload.stream) {
    return events(response)
  }

  return (await response.json()) as ResponsesResult
}
