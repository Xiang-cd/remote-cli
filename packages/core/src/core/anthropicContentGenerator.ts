/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * ContentGenerator adapter for the Anthropic Messages API.
 *
 * Translates between Gemini SDK types (used internally by the codebase) and
 * Anthropic's native message format, supporting text, tool use, and streaming.
 */

import type {
  GenerateContentParameters,
  GenerateContentResponse,
  CountTokensParameters,
  CountTokensResponse,
  EmbedContentParameters,
  EmbedContentResponse,
  Content,
  Part,
} from '@google/genai';
import type { ContentGenerator } from './contentGenerator.js';
import type { UserTierId, GeminiUserTier } from '../code_assist/types.js';
import type { LlmRole } from '../telemetry/llmRole.js';
import {
  geminiRequestToAnthropic,
  anthropicResponseToGemini,
  type AnthropicResponse,
  type AnthropicStreamEvent,
} from './formatConverters.js';
import { debugLogger } from '../utils/debugLogger.js';
import { estimateTokenCountSync } from '../utils/tokenCalculation.js';

interface AnthropicClientConfig {
  apiKey: string;
  baseURL?: string;
  defaultHeaders?: Record<string, string>;
}

const DEFAULT_ANTHROPIC_BASE_URL = 'https://api.anthropic.com';
const ANTHROPIC_API_VERSION = '2023-06-01';

export class AnthropicContentGenerator implements ContentGenerator {
  private readonly apiKey: string;
  private readonly baseURL: string;
  private readonly defaultHeaders: Record<string, string>;

  userTier?: UserTierId;
  userTierName?: string;
  paidTier?: GeminiUserTier;

  constructor(config: AnthropicClientConfig) {
    this.apiKey = config.apiKey;
    this.baseURL = (config.baseURL ?? DEFAULT_ANTHROPIC_BASE_URL).replace(
      /\/+$/,
      '',
    );
    this.defaultHeaders = config.defaultHeaders ?? {};
  }

  async generateContent(
    request: GenerateContentParameters,
    _userPromptId: string,
    _role: LlmRole,
  ): Promise<GenerateContentResponse> {
    const anthropicRequest = geminiRequestToAnthropic(request);

    const response = await this.fetchAnthropic('/v1/messages', {
      ...anthropicRequest,
      stream: false,
    });

    const json = (await response.json()) as AnthropicResponse;
    return anthropicResponseToGemini(json);
  }

  async generateContentStream(
    request: GenerateContentParameters,
    _userPromptId: string,
    _role: LlmRole,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    const anthropicRequest = geminiRequestToAnthropic(request);

    const response = await this.fetchAnthropic('/v1/messages', {
      ...anthropicRequest,
      stream: true,
    });

    if (!response.body) {
      throw new Error('Anthropic streaming response has no body');
    }

    return this.streamToAsyncGenerator(response.body);
  }

  async countTokens(
    request: CountTokensParameters,
  ): Promise<CountTokensResponse> {
    const contents = request.contents;
    let totalTokens = 0;

    if (contents) {
      const contentArray = Array.isArray(contents) ? contents : [contents];
      for (const content of contentArray) {
        // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
        const c = content as Content;
        if (c.parts) {
          totalTokens += estimateTokenCountSync(c.parts);
        }
      }
    }

    return { totalTokens };
  }

  async embedContent(
    _request: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    throw new Error(
      'Anthropic does not support embeddings. Use an OpenAI-compatible provider for embeddings.',
    );
  }

  // ---------------------------------------------------------------------------
  // Private helpers
  // ---------------------------------------------------------------------------

  private async fetchAnthropic(
    path: string,
    body: Record<string, unknown>,
  ): Promise<Response> {
    const url = `${this.baseURL}${path}`;

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'x-api-key': this.apiKey,
      'anthropic-version': ANTHROPIC_API_VERSION,
      ...this.defaultHeaders,
    };

    // Remove undefined values
    const cleanBody = Object.fromEntries(
      Object.entries(body).filter(([, v]) => v !== undefined),
    );

    const toolCount = Array.isArray(cleanBody['tools'])
      ? (cleanBody['tools'] as unknown[]).length
      : 0;
    debugLogger.debug(
      `[Anthropic] POST ${url} model=${String(cleanBody['model'])} tools=${toolCount} tool_choice=${JSON.stringify(cleanBody['tool_choice'] ?? 'unset')}`,
    );

    const response = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(cleanBody),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Anthropic API error ${response.status}: ${errorText}`);
    }

    return response;
  }

  private async *streamToAsyncGenerator(
    body: ReadableStream<Uint8Array>,
  ): AsyncGenerator<GenerateContentResponse> {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    // Accumulate tool use blocks across content_block_start + delta events
    const pendingToolUses = new Map<
      number,
      { id: string; name: string; partialJson: string }
    >();

    let inputTokens = 0;
    let outputTokens = 0;
    let responseId = '';
    let model = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          const trimmed = line.trim();

          if (trimmed.startsWith('data: ')) {
            const data = trimmed.slice(6);

            try {
              const event = JSON.parse(data) as AnthropicStreamEvent;

              switch (event.type) {
                case 'message_start': {
                  if (event.message) {
                    responseId = event.message.id;
                    model = event.message.model;
                    if (event.message.usage) {
                      inputTokens = event.message.usage.input_tokens;
                    }
                  }
                  break;
                }

                case 'content_block_start': {
                  if (
                    event.content_block?.type === 'tool_use' &&
                    event.index !== undefined
                  ) {
                    pendingToolUses.set(event.index, {
                      id: event.content_block.id ?? '',
                      name: event.content_block.name ?? '',
                      partialJson: '',
                    });
                  }
                  break;
                }

                case 'content_block_delta': {
                  if (!event.delta) break;

                  if (event.delta.type === 'text_delta' && event.delta.text) {
                    yield this.makeGeminiChunk(
                      [{ text: event.delta.text }],
                      undefined,
                      responseId,
                      model,
                    );
                  } else if (
                    event.delta.type === 'input_json_delta' &&
                    event.index !== undefined
                  ) {
                    const pending = pendingToolUses.get(event.index);
                    if (pending) {
                      pending.partialJson += event.delta.partial_json ?? '';
                    }
                  }
                  // Skip thinking_delta, signature_delta for now
                  break;
                }

                case 'content_block_stop': {
                  if (event.index !== undefined) {
                    const pending = pendingToolUses.get(event.index);
                    if (pending) {
                      let input: Record<string, unknown> = {};
                      try {
                        input = JSON.parse(
                          pending.partialJson || '{}',
                        ) as Record<string, unknown>;
                      } catch {
                        input = { _raw: pending.partialJson };
                      }
                      yield this.makeGeminiChunk(
                        [
                          {
                            functionCall: {
                              name: pending.name,
                              args: input,
                              id: pending.id,
                            },
                          },
                        ],
                        undefined,
                        responseId,
                        model,
                      );
                      pendingToolUses.delete(event.index);
                    }
                  }
                  break;
                }

                case 'message_delta': {
                  if (event.usage?.output_tokens) {
                    outputTokens = event.usage.output_tokens;
                  }
                  const finishReason = event.delta?.stop_reason
                    ? mapAnthropicStopReasonLocal(event.delta.stop_reason)
                    : undefined;

                  if (finishReason) {
                    yield this.makeGeminiChunk(
                      [],
                      finishReason,
                      responseId,
                      model,
                      {
                        promptTokenCount: inputTokens,
                        candidatesTokenCount: outputTokens,
                        totalTokenCount: inputTokens + outputTokens,
                      },
                    );
                  }
                  break;
                }

                case 'message_stop':
                  break;

                case 'ping':
                  break;

                case 'error': {
                  const errorData = event as unknown as {
                    error?: { type?: string; message?: string };
                  };
                  throw new Error(
                    `Anthropic stream error: ${errorData.error?.message ?? 'unknown'}`,
                  );
                }

                default:
                  break;
              }
            } catch (e) {
              if (e instanceof SyntaxError) {
                debugLogger.debug(
                  `[Anthropic] Failed to parse SSE data: ${data}`,
                );
              } else {
                throw e;
              }
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  private makeGeminiChunk(
    parts: Part[],
    finishReason: string | undefined,
    responseId: string,
    model: string,
    usage?: {
      promptTokenCount: number;
      candidatesTokenCount: number;
      totalTokenCount: number;
    },
  ): GenerateContentResponse {
    const geminiResponse = {
      candidates:
        parts.length > 0 || finishReason
          ? [
              {
                content: { role: 'model', parts },
                finishReason,
                index: 0,
              },
            ]
          : [],
      usageMetadata: usage,
      modelVersion: model,
      responseId,
    };

    return geminiResponse as unknown as GenerateContentResponse;
  }
}

function mapAnthropicStopReasonLocal(
  reason: string,
): string | undefined {
  switch (reason) {
    case 'end_turn':
      return 'STOP';
    case 'max_tokens':
      return 'MAX_TOKENS';
    case 'tool_use':
      return 'STOP';
    case 'stop_sequence':
      return 'STOP';
    default:
      return 'STOP';
  }
}
