/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Bidirectional format converters between Gemini SDK types and OpenAI API types.
 *
 * Gemini uses: Content[] with Parts (text, functionCall, functionResponse)
 * OpenAI uses: ChatCompletionMessageParam[] with tool_calls / tool role
 */

import type {
  Content,
  GenerateContentParameters,
  GenerateContentResponse,
  Part,
  Tool,
  FunctionDeclaration,
} from '@google/genai';

// ---------------------------------------------------------------------------
// OpenAI-compatible types (subset relevant to this adapter)
// ---------------------------------------------------------------------------

export interface OpenAIChatMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string | null;
  tool_calls?: OpenAIToolCall[];
  tool_call_id?: string;
}

export interface OpenAIToolCall {
  id: string;
  type: 'function';
  function: {
    name: string;
    arguments: string;
  };
}

export interface OpenAITool {
  type: 'function';
  function: {
    name: string;
    description?: string;
    parameters?: Record<string, unknown>;
  };
}

export interface OpenAIChatCompletionChoice {
  index: number;
  message: OpenAIChatMessage;
  finish_reason: string | null;
}

export interface OpenAIChatCompletionResponse {
  id: string;
  object: string;
  model: string;
  choices: OpenAIChatCompletionChoice[];
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

export interface OpenAIStreamDelta {
  role?: string;
  content?: string | null;
  tool_calls?: Array<{
    index: number;
    id?: string;
    type?: string;
    function?: {
      name?: string;
      arguments?: string;
    };
  }>;
}

export interface OpenAIStreamChoice {
  index: number;
  delta: OpenAIStreamDelta;
  finish_reason: string | null;
}

export interface OpenAIStreamChunk {
  id: string;
  object: string;
  model: string;
  choices: OpenAIStreamChoice[];
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

// ---------------------------------------------------------------------------
// Gemini -> OpenAI conversion
// ---------------------------------------------------------------------------

/**
 * Convert Gemini Content[] to OpenAI messages, prepending systemInstruction.
 */
export function geminiContentsToOpenAIMessages(
  contents: Content[] | undefined,
  systemInstruction?: Content | string,
): OpenAIChatMessage[] {
  const messages: OpenAIChatMessage[] = [];

  if (systemInstruction) {
    const text =
      typeof systemInstruction === 'string'
        ? systemInstruction
        : systemInstruction.parts
            ?.map((p: Part) => p.text ?? '')
            .filter(Boolean)
            .join('\n') ?? '';
    if (text) {
      messages.push({ role: 'system', content: text });
    }
  }

  if (!contents) return messages;

  for (const content of contents) {
    const role = content.role === 'model' ? 'assistant' : 'user';
    const parts = content.parts ?? [];

    const textParts: string[] = [];
    const toolCalls: OpenAIToolCall[] = [];
    const functionResponses: Array<{
      id: string;
      name: string;
      content: string;
    }> = [];

    for (const part of parts) {
      if (part.text !== undefined) {
        textParts.push(part.text);
      } else if (part.functionCall) {
        toolCalls.push({
          id: part.functionCall.id ?? `call_${part.functionCall.name}`,
          type: 'function',
          function: {
            name: part.functionCall.name ?? '',
            arguments: JSON.stringify(part.functionCall.args ?? {}),
          },
        });
      } else if (part.functionResponse) {
        functionResponses.push({
          id: part.functionResponse.id ?? `call_${part.functionResponse.name}`,
          name: part.functionResponse.name ?? '',
          content: typeof part.functionResponse.response === 'string'
            ? part.functionResponse.response
            : JSON.stringify(part.functionResponse.response ?? {}),
        });
      }
      // Skip thought, thoughtSignature, inlineData, etc. for OpenAI
    }

    if (toolCalls.length > 0) {
      // Assistant message with tool calls
      messages.push({
        role: 'assistant',
        content: textParts.length > 0 ? textParts.join('') : null,
        tool_calls: toolCalls,
      });
    } else if (functionResponses.length > 0) {
      // Each function response becomes a separate tool message
      for (const fr of functionResponses) {
        messages.push({
          role: 'tool',
          tool_call_id: fr.id,
          content: fr.content,
        });
      }
    } else {
      messages.push({
        role,
        content: textParts.join('') || '',
      });
    }
  }

  return messages;
}

/**
 * Convert Gemini Tool[] (with functionDeclarations) to OpenAI tools format.
 */
export function geminiToolsToOpenAI(
  tools: Tool[] | undefined,
): OpenAITool[] | undefined {
  if (!tools || tools.length === 0) return undefined;

  const openAITools: OpenAITool[] = [];
  for (const tool of tools) {
    if (
      tool &&
      typeof tool === 'object' &&
      'functionDeclarations' in tool &&
      tool.functionDeclarations
    ) {
      for (const decl of tool.functionDeclarations as FunctionDeclaration[]) {
        openAITools.push({
          type: 'function',
          function: {
            name: decl.name ?? '',
            description: decl.description,
            parameters:
              // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
              ((decl as Record<string, unknown>)['parametersJsonSchema'] as
                | Record<string, unknown>
                | undefined) ??
              (decl.parameters as Record<string, unknown> | undefined),
          },
        });
      }
    }
  }

  return openAITools.length > 0 ? openAITools : undefined;
}

/**
 * Build the full OpenAI request body from Gemini GenerateContentParameters.
 */
export function geminiRequestToOpenAI(
  request: GenerateContentParameters,
): {
  model: string;
  messages: OpenAIChatMessage[];
  tools?: OpenAITool[];
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  stop?: string[];
  response_format?: { type: string };
} {
  const config = request.config;

  // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
  const systemInstruction = config?.systemInstruction as
    | Content
    | string
    | undefined;
  // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
  const tools = config?.tools as Tool[] | undefined;

  const result: ReturnType<typeof geminiRequestToOpenAI> = {
    model: request.model ?? '',
    messages: geminiContentsToOpenAIMessages(
      // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
      request.contents as Content[] | undefined,
      systemInstruction,
    ),
    tools: geminiToolsToOpenAI(tools),
    temperature: config?.temperature,
    top_p: config?.topP,
  };

  if (config?.maxOutputTokens) {
    result.max_tokens = config.maxOutputTokens;
  }

  if (config?.responseMimeType === 'application/json') {
    result.response_format = { type: 'json_object' };
  }

  return result;
}

// ---------------------------------------------------------------------------
// OpenAI -> Gemini conversion (responses)
// ---------------------------------------------------------------------------

/**
 * Convert an OpenAI ChatCompletion response to Gemini GenerateContentResponse shape.
 */
export function openAIResponseToGemini(
  response: OpenAIChatCompletionResponse,
): GenerateContentResponse {
  const choice = response.choices?.[0];
  const parts: Part[] = [];

  if (choice?.message) {
    if (choice.message.content) {
      parts.push({ text: choice.message.content });
    }
    if (choice.message.tool_calls) {
      for (const tc of choice.message.tool_calls) {
        let args: Record<string, unknown> = {};
        try {
          args = JSON.parse(tc.function.arguments) as Record<string, unknown>;
        } catch {
          args = { _raw: tc.function.arguments };
        }
        parts.push({
          functionCall: {
            name: tc.function.name,
            args,
            id: tc.id,
          },
        });
      }
    }
  }

  const finishReason = mapOpenAIFinishReason(choice?.finish_reason);

  // Construct a plain object matching GenerateContentResponse shape
  const geminiResponse = {
    candidates: [
      {
        content: {
          role: 'model',
          parts,
        },
        finishReason,
        index: 0,
      },
    ],
    usageMetadata: response.usage
      ? {
          promptTokenCount: response.usage.prompt_tokens,
          candidatesTokenCount: response.usage.completion_tokens,
          totalTokenCount: response.usage.total_tokens,
        }
      : undefined,
    modelVersion: response.model,
    responseId: response.id,
  };

  return geminiResponse as unknown as GenerateContentResponse;
}

/**
 * Convert an OpenAI streaming chunk to a Gemini GenerateContentResponse.
 * Streaming chunks have deltas rather than complete messages.
 */
export function openAIStreamChunkToGemini(
  chunk: OpenAIStreamChunk,
): GenerateContentResponse {
  const choice = chunk.choices?.[0];
  const parts: Part[] = [];

  if (choice?.delta) {
    if (choice.delta.content) {
      parts.push({ text: choice.delta.content });
    }
    if (choice.delta.tool_calls) {
      for (const tc of choice.delta.tool_calls) {
        if (tc.function?.name || tc.function?.arguments) {
          let args: Record<string, unknown> = {};
          if (tc.function.arguments) {
            try {
              args = JSON.parse(tc.function.arguments) as Record<
                string,
                unknown
              >;
            } catch {
              args = { _raw: tc.function.arguments };
            }
          }
          parts.push({
            functionCall: {
              name: tc.function.name ?? '',
              args,
              id: tc.id,
            },
          });
        }
      }
    }
  }

  const finishReason = mapOpenAIFinishReason(choice?.finish_reason);

  const geminiResponse = {
    candidates:
      parts.length > 0 || finishReason
        ? [
            {
              content: {
                role: 'model',
                parts,
              },
              finishReason,
              index: 0,
            },
          ]
        : [],
    usageMetadata: chunk.usage
      ? {
          promptTokenCount: chunk.usage.prompt_tokens,
          candidatesTokenCount: chunk.usage.completion_tokens,
          totalTokenCount: chunk.usage.total_tokens,
        }
      : undefined,
    modelVersion: chunk.model,
    responseId: chunk.id,
  };

  return geminiResponse as unknown as GenerateContentResponse;
}

// ---------------------------------------------------------------------------
// Anthropic-compatible types
// ---------------------------------------------------------------------------

export interface AnthropicContentBlock {
  type: 'text' | 'tool_use' | 'tool_result' | 'thinking';
  text?: string;
  id?: string;
  name?: string;
  input?: Record<string, unknown>;
  tool_use_id?: string;
  content?: string | AnthropicContentBlock[];
  thinking?: string;
  signature?: string;
}

export interface AnthropicMessage {
  role: 'user' | 'assistant';
  content: string | AnthropicContentBlock[];
}

export interface AnthropicTool {
  name: string;
  description?: string;
  input_schema: Record<string, unknown>;
}

export interface AnthropicResponse {
  id: string;
  type: 'message';
  role: 'assistant';
  model: string;
  content: AnthropicContentBlock[];
  stop_reason: string | null;
  stop_sequence: string | null;
  usage: {
    input_tokens: number;
    output_tokens: number;
  };
}

export interface AnthropicStreamEvent {
  type: string;
  index?: number;
  message?: AnthropicResponse;
  content_block?: AnthropicContentBlock;
  delta?: {
    type?: string;
    text?: string;
    partial_json?: string;
    stop_reason?: string;
    thinking?: string;
    signature?: string;
  };
  usage?: {
    input_tokens?: number;
    output_tokens?: number;
  };
}

// ---------------------------------------------------------------------------
// Gemini -> Anthropic conversion
// ---------------------------------------------------------------------------

/**
 * Convert Gemini Content[] to Anthropic messages format.
 * Anthropic requires system prompt as separate top-level param, not in messages.
 * Returns { system, messages }.
 */
export function geminiContentsToAnthropicMessages(
  contents: Content[] | undefined,
  systemInstruction?: Content | string,
): { system?: string; messages: AnthropicMessage[] } {
  let system: string | undefined;

  if (systemInstruction) {
    system =
      typeof systemInstruction === 'string'
        ? systemInstruction
        : systemInstruction.parts
            ?.map((p: Part) => p.text ?? '')
            .filter(Boolean)
            .join('\n') ?? '';
    if (!system) system = undefined;
  }

  const messages: AnthropicMessage[] = [];
  if (!contents) return { system, messages };

  for (const content of contents) {
    const role: 'user' | 'assistant' =
      content.role === 'model' ? 'assistant' : 'user';
    const parts = content.parts ?? [];

    const contentBlocks: AnthropicContentBlock[] = [];
    const toolResults: AnthropicContentBlock[] = [];

    for (const part of parts) {
      if (part.text !== undefined && part.text !== '') {
        contentBlocks.push({ type: 'text', text: part.text });
      } else if (part.functionCall) {
        contentBlocks.push({
          type: 'tool_use',
          id: part.functionCall.id ?? `toolu_${part.functionCall.name}`,
          name: part.functionCall.name ?? '',
          input: (part.functionCall.args as Record<string, unknown>) ?? {},
        });
      } else if (part.functionResponse) {
        const responseContent =
          typeof part.functionResponse.response === 'string'
            ? part.functionResponse.response
            : JSON.stringify(part.functionResponse.response ?? {});
        toolResults.push({
          type: 'tool_result',
          tool_use_id:
            part.functionResponse.id ??
            `toolu_${part.functionResponse.name}`,
          content: responseContent,
        });
      }
      // Skip thought, thoughtSignature, inlineData for Anthropic
    }

    if (toolResults.length > 0) {
      // tool_result blocks must go in a user message
      messages.push({ role: 'user', content: toolResults });
    } else if (contentBlocks.length > 0) {
      messages.push({ role, content: contentBlocks });
    } else {
      messages.push({ role, content: '' });
    }
  }

  // Anthropic requires alternating user/assistant messages.
  // Merge consecutive same-role messages.
  return { system, messages: mergeConsecutiveMessages(messages) };
}

function mergeConsecutiveMessages(
  messages: AnthropicMessage[],
): AnthropicMessage[] {
  if (messages.length <= 1) return messages;

  const merged: AnthropicMessage[] = [messages[0]!];
  for (let i = 1; i < messages.length; i++) {
    const current = messages[i]!;
    const last = merged[merged.length - 1]!;

    if (current.role === last.role) {
      // Merge content blocks
      const lastBlocks = toContentBlocks(last.content);
      const currentBlocks = toContentBlocks(current.content);
      last.content = [...lastBlocks, ...currentBlocks];
    } else {
      merged.push(current);
    }
  }

  return merged;
}

function toContentBlocks(
  content: string | AnthropicContentBlock[],
): AnthropicContentBlock[] {
  if (typeof content === 'string') {
    return content ? [{ type: 'text', text: content }] : [];
  }
  return content;
}

/**
 * Convert Gemini Tool[] to Anthropic tools format.
 */
export function geminiToolsToAnthropic(
  tools: Tool[] | undefined,
): AnthropicTool[] | undefined {
  if (!tools || tools.length === 0) return undefined;

  const anthropicTools: AnthropicTool[] = [];
  for (const tool of tools) {
    if (
      tool &&
      typeof tool === 'object' &&
      'functionDeclarations' in tool &&
      tool.functionDeclarations
    ) {
      for (const decl of tool.functionDeclarations as FunctionDeclaration[]) {
        const schema =
          // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
          ((decl as Record<string, unknown>)['parametersJsonSchema'] as
            | Record<string, unknown>
            | undefined) ??
          (decl.parameters as Record<string, unknown> | undefined) ??
          { type: 'object', properties: {} };

        anthropicTools.push({
          name: decl.name ?? '',
          description: decl.description,
          input_schema: schema,
        });
      }
    }
  }

  return anthropicTools.length > 0 ? anthropicTools : undefined;
}

/**
 * Build the full Anthropic request body from Gemini GenerateContentParameters.
 */
export function geminiRequestToAnthropic(
  request: GenerateContentParameters,
): {
  model: string;
  messages: AnthropicMessage[];
  system?: string;
  tools?: AnthropicTool[];
  max_tokens: number;
  temperature?: number;
  top_p?: number;
  top_k?: number;
} {
  const config = request.config;

  // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
  const systemInstruction = config?.systemInstruction as
    | Content
    | string
    | undefined;
  // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
  const tools = config?.tools as Tool[] | undefined;

  const { system, messages } = geminiContentsToAnthropicMessages(
    // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
    request.contents as Content[] | undefined,
    systemInstruction,
  );

  return {
    model: request.model ?? '',
    messages,
    system,
    tools: geminiToolsToAnthropic(tools),
    max_tokens: config?.maxOutputTokens ?? 8192,
    temperature: config?.temperature,
    top_p: config?.topP,
    top_k: config?.topK,
  };
}

// ---------------------------------------------------------------------------
// Anthropic -> Gemini conversion (responses)
// ---------------------------------------------------------------------------

/**
 * Convert an Anthropic response to Gemini GenerateContentResponse shape.
 */
export function anthropicResponseToGemini(
  response: AnthropicResponse,
): GenerateContentResponse {
  const parts: Part[] = [];

  for (const block of response.content) {
    if (block.type === 'text' && block.text) {
      parts.push({ text: block.text });
    } else if (block.type === 'tool_use') {
      parts.push({
        functionCall: {
          name: block.name ?? '',
          args: block.input ?? {},
          id: block.id,
        },
      });
    }
  }

  const finishReason = mapAnthropicStopReason(response.stop_reason);

  const geminiResponse = {
    candidates: [
      {
        content: { role: 'model', parts },
        finishReason,
        index: 0,
      },
    ],
    usageMetadata: {
      promptTokenCount: response.usage.input_tokens,
      candidatesTokenCount: response.usage.output_tokens,
      totalTokenCount:
        response.usage.input_tokens + response.usage.output_tokens,
    },
    modelVersion: response.model,
    responseId: response.id,
  };

  return geminiResponse as unknown as GenerateContentResponse;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function mapOpenAIFinishReason(
  reason: string | null | undefined,
): string | undefined {
  if (!reason) return undefined;
  switch (reason) {
    case 'stop':
      return 'STOP';
    case 'length':
      return 'MAX_TOKENS';
    case 'tool_calls':
      return 'STOP';
    case 'content_filter':
      return 'SAFETY';
    default:
      return 'STOP';
  }
}

function mapAnthropicStopReason(
  reason: string | null | undefined,
): string | undefined {
  if (!reason) return undefined;
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
