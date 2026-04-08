// Bridge configuration

export type BridgeDirection = "anthropic-to-openai" | "openai-to-anthropic"

export interface BridgeConfig {
  port: number
  apiKey: string
  direction: BridgeDirection
  targetBaseUrl: string
  targetApiKey: string
  verbose: boolean
}

export function getConfig(): BridgeConfig {
  const direction = (process.env.BRIDGE_DIRECTION || "anthropic-to-openai") as BridgeDirection

  let targetBaseUrl: string
  if (process.env.TARGET_BASE_URL) {
    targetBaseUrl = process.env.TARGET_BASE_URL
  } else if (direction === "anthropic-to-openai") {
    targetBaseUrl = "https://api.openai.com"
  } else {
    targetBaseUrl = "https://api.anthropic.com"
  }

  return {
    port: Number(process.env.PORT || 4141),
    apiKey: process.env.API_KEY || "",
    direction,
    targetBaseUrl,
    targetApiKey: process.env.TARGET_API_KEY || "",
    verbose: process.env.VERBOSE === "true",
  }
}
