/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * ContentGenerator adapter for OpenAI-compatible APIs.
 *
 * Translates between Gemini SDK types (used internally by the codebase) and
 * OpenAI chat completion format. Works with any OpenAI-compatible endpoint
 * including OpenAI, OpenRouter, one-api, LiteLLM proxy, Azure OpenAI, etc.
 */

import type {
  GenerateContentParameters,
  GenerateContentResponse,
  CountTokensParameters,
  CountTokensResponse,
  EmbedContentParameters,
  EmbedContentResponse,
  Content,
} from '@google/genai';
import type { ContentGenerator } from './contentGenerator.js';
import type { UserTierId, GeminiUserTier } from '../code_assist/types.js';
import type { LlmRole } from '../telemetry/llmRole.js';
import {
  geminiRequestToOpenAI,
  openAIResponseToGemini,
  openAIStreamChunkToGemini,
  type OpenAIChatCompletionResponse,
  type OpenAIStreamChunk,
} from './formatConverters.js';
import { debugLogger } from '../utils/debugLogger.js';
import { estimateTokenCountSync } from '../utils/tokenCalculation.js';

interface OpenAIClientConfig {
  apiKey: string;
  baseURL: string;
  defaultHeaders?: Record<string, string>;
}

export class OpenAIContentGenerator implements ContentGenerator {
  private readonly apiKey: string;
  private readonly baseURL: string;
  private readonly defaultHeaders: Record<string, string>;

  userTier?: UserTierId;
  userTierName?: string;
  paidTier?: GeminiUserTier;

  constructor(config: OpenAIClientConfig) {
    this.apiKey = config.apiKey;
    this.baseURL = config.baseURL.replace(/\/+$/, '');
    this.defaultHeaders = config.defaultHeaders ?? {};
  }

  async generateContent(
    request: GenerateContentParameters,
    _userPromptId: string,
    _role: LlmRole,
  ): Promise<GenerateContentResponse> {
    const openAIRequest = geminiRequestToOpenAI(request);

    const response = await this.fetchOpenAI('/chat/completions', {
      ...openAIRequest,
      stream: false,
    });

    const json = (await response.json()) as OpenAIChatCompletionResponse;
    return openAIResponseToGemini(json);
  }

  async generateContentStream(
    request: GenerateContentParameters,
    _userPromptId: string,
    _role: LlmRole,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    const openAIRequest = geminiRequestToOpenAI(request);

    const response = await this.fetchOpenAI('/chat/completions', {
      ...openAIRequest,
      stream: true,
      stream_options: { include_usage: true },
    });

    if (!response.body) {
      throw new Error('OpenAI streaming response has no body');
    }

    return this.streamToAsyncGenerator(response.body, request);
  }

  async countTokens(
    request: CountTokensParameters,
  ): Promise<CountTokensResponse> {
    // OpenAI doesn't have a direct countTokens endpoint.
    // Provide an estimate based on character count heuristic.
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

    return {
      totalTokens,
    };
  }

  async embedContent(
    request: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    // Map to OpenAI embeddings endpoint
    const contents = request.contents;
    let inputText = '';

    if (typeof contents === 'string') {
      inputText = contents;
    } else if (Array.isArray(contents)) {
      inputText = contents
        .map((c) => {
          if (typeof c === 'string') return c;
          // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
          const content = c as Content;
          return content.parts?.map((p: { text?: string }) => p.text ?? '').join('') ?? '';
        })
        .join('\n');
    }

    const response = await this.fetchOpenAI('/embeddings', {
      model: request.model ?? 'text-embedding-ada-002',
      input: inputText,
    });

    const json = (await response.json()) as {
      data: Array<{ embedding: number[] }>;
    };

    return {
      embedding: {
        values: json.data?.[0]?.embedding ?? [],
      },
    } as unknown as EmbedContentResponse;
  }

  // ---------------------------------------------------------------------------
  // Private helpers
  // ---------------------------------------------------------------------------

  private async fetchOpenAI(
    path: string,
    body: Record<string, unknown>,
  ): Promise<Response> {
    const url = `${this.baseURL}${path}`;

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${this.apiKey}`,
      ...this.defaultHeaders,
    };

    const abortSignal =
      body['stream'] === false
        ? // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
          (body['abort_signal'] as AbortSignal | undefined)
        : undefined;

    // Remove non-serializable fields
    const { abort_signal: _, ...serializableBody } = body as Record<
      string,
      unknown
    > & { abort_signal?: unknown };

    // Remove undefined values to keep the request clean
    const cleanBody = Object.fromEntries(
      Object.entries(serializableBody).filter(([, v]) => v !== undefined),
    );

    debugLogger.debug(
      `[OpenAI] POST ${url} model=${String(cleanBody['model'])}`,
    );

    const response = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(cleanBody),
      signal: abortSignal,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(
        `OpenAI API error ${response.status}: ${errorText}`,
      );
    }

    return response;
  }

  private async *streamToAsyncGenerator(
    body: ReadableStream<Uint8Array>,
    _request: GenerateContentParameters,
  ): AsyncGenerator<GenerateContentResponse> {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    // Accumulated tool call state for streaming assembly
    const pendingToolCalls = new Map<
      number,
      { id: string; name: string; arguments: string }
    >();

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || !trimmed.startsWith('data: ')) continue;

          const data = trimmed.slice(6);
          if (data === '[DONE]') {
            // Flush any accumulated tool calls as a final chunk
            if (pendingToolCalls.size > 0) {
              yield this.flushPendingToolCalls(pendingToolCalls);
              pendingToolCalls.clear();
            }
            return;
          }

          try {
            const chunk = JSON.parse(data) as OpenAIStreamChunk;
            const delta = chunk.choices?.[0]?.delta;

            // Accumulate tool calls across chunks since OpenAI streams
            // them in fragments (name in one chunk, arguments spread across many)
            if (delta?.tool_calls) {
              for (const tc of delta.tool_calls) {
                const existing = pendingToolCalls.get(tc.index);
                if (existing) {
                  if (tc.function?.arguments) {
                    existing.arguments += tc.function.arguments;
                  }
                } else {
                  pendingToolCalls.set(tc.index, {
                    id: tc.id ?? '',
                    name: tc.function?.name ?? '',
                    arguments: tc.function?.arguments ?? '',
                  });
                }
              }

              // If finish_reason is set, flush accumulated tool calls
              if (chunk.choices?.[0]?.finish_reason) {
                yield this.flushPendingToolCalls(pendingToolCalls, chunk);
                pendingToolCalls.clear();
              }
              continue;
            }

            // For text content, yield directly
            if (delta?.content) {
              yield openAIStreamChunkToGemini(chunk);
            }

            // Handle finish_reason without content (e.g., end of text)
            if (
              chunk.choices?.[0]?.finish_reason &&
              !delta?.content &&
              !delta?.tool_calls
            ) {
              // Yield usage metadata if present
              if (chunk.usage) {
                yield openAIStreamChunkToGemini(chunk);
              }
            }
          } catch (e) {
            debugLogger.debug(`[OpenAI] Failed to parse SSE chunk: ${data}`, e);
          }
        }
      }

      // Flush any remaining tool calls
      if (pendingToolCalls.size > 0) {
        yield this.flushPendingToolCalls(pendingToolCalls);
        pendingToolCalls.clear();
      }
    } finally {
      reader.releaseLock();
    }
  }

  /**
   * Assemble accumulated tool call fragments into a single Gemini response.
   */
  private flushPendingToolCalls(
    pending: Map<number, { id: string; name: string; arguments: string }>,
    chunk?: OpenAIStreamChunk,
  ): GenerateContentResponse {
    const parts = Array.from(pending.values()).map((tc) => {
      let args: Record<string, unknown> = {};
      try {
        args = JSON.parse(tc.arguments) as Record<string, unknown>;
      } catch {
        args = { _raw: tc.arguments };
      }
      return {
        functionCall: {
          name: tc.name,
          args,
          id: tc.id,
        },
      };
    });

    const geminiResponse = {
      candidates: [
        {
          content: {
            role: 'model',
            parts,
          },
          finishReason: 'STOP',
          index: 0,
        },
      ],
      usageMetadata: chunk?.usage
        ? {
            promptTokenCount: chunk.usage.prompt_tokens,
            candidatesTokenCount: chunk.usage.completion_tokens,
            totalTokenCount: chunk.usage.total_tokens,
          }
        : undefined,
      modelVersion: chunk?.model,
      responseId: chunk?.id,
    };

    return geminiResponse as unknown as GenerateContentResponse;
  }
}
