/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type {
  CountTokensResponse,
  GenerateContentResponse,
  GenerateContentParameters,
  CountTokensParameters,
  EmbedContentResponse,
  EmbedContentParameters,
} from '@google/genai';
import { GoogleGenAI } from '@google/genai';
import { createCodeAssistContentGenerator } from '../code_assist/codeAssist.js';
import type { Config } from '../config/config.js';
import { loadApiKey } from './apiKeyCredentialStorage.js';

import type { UserTierId, GeminiUserTier } from '../code_assist/types.js';
import { LoggingContentGenerator } from './loggingContentGenerator.js';
import { InstallationManager } from '../utils/installationManager.js';
import { FakeContentGenerator } from './fakeContentGenerator.js';
import { parseCustomHeaders } from '../utils/customHeaderUtils.js';
import { RecordingContentGenerator } from './recordingContentGenerator.js';
import { OpenAIContentGenerator } from './openaiContentGenerator.js';
import { AnthropicContentGenerator } from './anthropicContentGenerator.js';
import { getVersion, resolveModel } from '../../index.js';
import type { LlmRole } from '../telemetry/llmRole.js';

/**
 * Interface abstracting the core functionalities for generating content and counting tokens.
 */
export interface ContentGenerator {
  generateContent(
    request: GenerateContentParameters,
    userPromptId: string,
    role: LlmRole,
  ): Promise<GenerateContentResponse>;

  generateContentStream(
    request: GenerateContentParameters,
    userPromptId: string,
    role: LlmRole,
  ): Promise<AsyncGenerator<GenerateContentResponse>>;

  countTokens(request: CountTokensParameters): Promise<CountTokensResponse>;

  embedContent(request: EmbedContentParameters): Promise<EmbedContentResponse>;

  userTier?: UserTierId;

  userTierName?: string;

  paidTier?: GeminiUserTier;
}

export enum AuthType {
  LOGIN_WITH_GOOGLE = 'oauth-personal',
  USE_GEMINI = 'gemini-api-key',
  USE_VERTEX_AI = 'vertex-ai',
  LEGACY_CLOUD_SHELL = 'cloud-shell',
  COMPUTE_ADC = 'compute-default-credentials',
}

/**
 * Detects the best authentication type based on environment variables.
 *
 * Checks in order:
 * 1. GOOGLE_GENAI_USE_GCA=true -> LOGIN_WITH_GOOGLE
 * 2. GOOGLE_GENAI_USE_VERTEXAI=true -> USE_VERTEX_AI
 * 3. GEMINI_API_KEY -> USE_GEMINI
 */
export function getAuthTypeFromEnv(): AuthType | undefined {
  if (process.env['GOOGLE_GENAI_USE_GCA'] === 'true') {
    return AuthType.LOGIN_WITH_GOOGLE;
  }
  if (process.env['GOOGLE_GENAI_USE_VERTEXAI'] === 'true') {
    return AuthType.USE_VERTEX_AI;
  }
  if (process.env['GEMINI_API_KEY']) {
    return AuthType.USE_GEMINI;
  }
  if (
    process.env['CLOUD_SHELL'] === 'true' ||
    process.env['GEMINI_CLI_USE_COMPUTE_ADC'] === 'true'
  ) {
    return AuthType.COMPUTE_ADC;
  }
  return undefined;
}

export type ContentGeneratorConfig = {
  apiKey?: string;
  vertexai?: boolean;
  authType?: AuthType;
  proxy?: string;
};

export async function createContentGeneratorConfig(
  config: Config,
  authType: AuthType | undefined,
  apiKey?: string,
): Promise<ContentGeneratorConfig> {
  const geminiApiKey =
    apiKey ||
    process.env['GEMINI_API_KEY'] ||
    (await loadApiKey()) ||
    undefined;
  const googleApiKey = process.env['GOOGLE_API_KEY'] || undefined;
  const googleCloudProject =
    process.env['GOOGLE_CLOUD_PROJECT'] ||
    process.env['GOOGLE_CLOUD_PROJECT_ID'] ||
    undefined;
  const googleCloudLocation = process.env['GOOGLE_CLOUD_LOCATION'] || undefined;

  const contentGeneratorConfig: ContentGeneratorConfig = {
    authType,
    proxy: config?.getProxy(),
  };

  // If we are using Google auth or we are in Cloud Shell, there is nothing else to validate for now
  if (
    authType === AuthType.LOGIN_WITH_GOOGLE ||
    authType === AuthType.COMPUTE_ADC
  ) {
    return contentGeneratorConfig;
  }

  if (authType === AuthType.USE_GEMINI && geminiApiKey) {
    contentGeneratorConfig.apiKey = geminiApiKey;
    contentGeneratorConfig.vertexai = false;

    return contentGeneratorConfig;
  }

  if (
    authType === AuthType.USE_VERTEX_AI &&
    (googleApiKey || (googleCloudProject && googleCloudLocation))
  ) {
    contentGeneratorConfig.apiKey = googleApiKey;
    contentGeneratorConfig.vertexai = true;

    return contentGeneratorConfig;
  }

  return contentGeneratorConfig;
}

export type ApiProvider = 'gemini' | 'openai' | 'anthropic';

export function getApiProvider(): ApiProvider {
  const provider = process.env['GEMINI_CLI_API_PROVIDER']?.toLowerCase();
  if (provider === 'openai' || provider === 'anthropic') {
    return provider;
  }
  return 'gemini';
}

export interface ApiModelInfo {
  id: string;
  owned_by?: string;
}

/**
 * Fetches available models from the configured API provider's /models endpoint.
 * Works with OpenAI-compatible and Anthropic APIs.
 */
export async function listModelsFromApi(): Promise<ApiModelInfo[]> {
  const provider = getApiProvider();
  const baseURL = process.env['GEMINI_CLI_API_BASE_URL'];
  const apiKey = process.env['GEMINI_API_KEY'];

  if (provider === 'gemini' || !apiKey) {
    return [];
  }

  const base = (baseURL ?? '').replace(/\/+$/, '');

  const headers: Record<string, string> = {};
  let modelsUrl: string;

  if (provider === 'anthropic') {
    headers['x-api-key'] = apiKey;
    headers['anthropic-version'] = '2023-06-01';
    modelsUrl = base ? `${base}/v1/models` : 'https://api.anthropic.com/v1/models';
  } else {
    headers['Authorization'] = `Bearer ${apiKey}`;
    modelsUrl = base ? `${base}/models` : 'https://api.openai.com/v1/models';
  }

  try {
    const response = await fetch(modelsUrl, { headers });
    if (!response.ok) {
      return [];
    }
    const json = (await response.json()) as {
      data?: Array<{ id: string; owned_by?: string }>;
    };
    const models = (json.data ?? [])
      .map((m) => ({ id: m.id, owned_by: m.owned_by }))
      .sort((a, b) => a.id.localeCompare(b.id));
    return models;
  } catch {
    return [];
  }
}

export async function createContentGenerator(
  config: ContentGeneratorConfig,
  gcConfig: Config,
  sessionId?: string,
): Promise<ContentGenerator> {
  const generator = await (async () => {
    if (gcConfig.fakeResponses) {
      const fakeGenerator = await FakeContentGenerator.fromFile(
        gcConfig.fakeResponses,
      );
      return new LoggingContentGenerator(fakeGenerator, gcConfig);
    }

    const provider = getApiProvider();

    // Route to OpenAI-compatible adapter when configured
    if (provider === 'openai') {
      return createOpenAIGenerator(config, gcConfig);
    }

    // Route to native Anthropic adapter
    if (provider === 'anthropic') {
      return createAnthropicGenerator(config, gcConfig);
    }

    // Default: Gemini provider (existing logic)
    return createGeminiGenerator(config, gcConfig, sessionId);
  })();

  if (gcConfig.recordResponses) {
    return new RecordingContentGenerator(generator, gcConfig.recordResponses);
  }

  return generator;
}

async function createOpenAIGenerator(
  config: ContentGeneratorConfig,
  gcConfig: Config,
): Promise<ContentGenerator> {
  const baseURL = process.env['GEMINI_CLI_API_BASE_URL'];
  if (!baseURL) {
    throw new Error(
      'GEMINI_CLI_API_BASE_URL is required when GEMINI_CLI_API_PROVIDER=openai. ' +
        'Example: GEMINI_CLI_API_BASE_URL=https://api.openai.com/v1',
    );
  }

  const apiKey = config.apiKey || process.env['GEMINI_API_KEY'];
  if (!apiKey) {
    throw new Error(
      'GEMINI_API_KEY is required when GEMINI_CLI_API_PROVIDER=openai.',
    );
  }

  const version = await getVersion();
  const model = gcConfig.getModel();
  const customHeadersEnv =
    process.env['GEMINI_CLI_CUSTOM_HEADERS'] || undefined;
  const customHeadersMap = parseCustomHeaders(customHeadersEnv);

  const openaiGenerator = new OpenAIContentGenerator({
    apiKey,
    baseURL,
    defaultHeaders: {
      ...customHeadersMap,
      'User-Agent': `GeminiCLI/${version}/${model} (${process.platform}; ${process.arch})`,
    },
  });

  return new LoggingContentGenerator(openaiGenerator, gcConfig);
}

async function createGeminiGenerator(
  config: ContentGeneratorConfig,
  gcConfig: Config,
  sessionId?: string,
): Promise<ContentGenerator> {
  const version = await getVersion();
  const model = resolveModel(
    gcConfig.getModel(),
    config.authType === AuthType.USE_GEMINI ||
      config.authType === AuthType.USE_VERTEX_AI ||
      ((await gcConfig.getGemini31Launched?.()) ?? false),
  );
  const customHeadersEnv =
    process.env['GEMINI_CLI_CUSTOM_HEADERS'] || undefined;
  const userAgent = `GeminiCLI/${version}/${model} (${process.platform}; ${process.arch})`;
  const customHeadersMap = parseCustomHeaders(customHeadersEnv);
  const apiKeyAuthMechanism =
    process.env['GEMINI_API_KEY_AUTH_MECHANISM'] || 'x-goog-api-key';
  const apiVersionEnv = process.env['GOOGLE_GENAI_API_VERSION'];

  const baseHeaders: Record<string, string> = {
    ...customHeadersMap,
    'User-Agent': userAgent,
  };

  if (
    apiKeyAuthMechanism === 'bearer' &&
    (config.authType === AuthType.USE_GEMINI ||
      config.authType === AuthType.USE_VERTEX_AI) &&
    config.apiKey
  ) {
    baseHeaders['Authorization'] = `Bearer ${config.apiKey}`;
  }
  if (
    config.authType === AuthType.LOGIN_WITH_GOOGLE ||
    config.authType === AuthType.COMPUTE_ADC
  ) {
    const httpOptions = { headers: baseHeaders };
    return new LoggingContentGenerator(
      await createCodeAssistContentGenerator(
        httpOptions,
        config.authType,
        gcConfig,
        sessionId,
      ),
      gcConfig,
    );
  }

  if (
    config.authType === AuthType.USE_GEMINI ||
    config.authType === AuthType.USE_VERTEX_AI
  ) {
    let headers: Record<string, string> = { ...baseHeaders };
    if (gcConfig?.getUsageStatisticsEnabled()) {
      const installationManager = new InstallationManager();
      const installationId = installationManager.getInstallationId();
      headers = {
        ...headers,
        'x-gemini-api-privileged-user-id': `${installationId}`,
      };
    }
    const customBaseUrl = process.env['GEMINI_CLI_API_BASE_URL'] || undefined;
    const httpOptions: Record<string, unknown> = { headers };
    if (customBaseUrl) {
      httpOptions['baseUrl'] = customBaseUrl;
    }

    const googleGenAI = new GoogleGenAI({
      apiKey: config.apiKey === '' ? undefined : config.apiKey,
      vertexai: config.vertexai,
      httpOptions,
      ...(apiVersionEnv && { apiVersion: apiVersionEnv }),
    });
    return new LoggingContentGenerator(googleGenAI.models, gcConfig);
  }
  throw new Error(
    `Error creating contentGenerator: Unsupported authType: ${config.authType}`,
  );
}

async function createAnthropicGenerator(
  config: ContentGeneratorConfig,
  gcConfig: Config,
): Promise<ContentGenerator> {
  const baseURL = process.env['GEMINI_CLI_API_BASE_URL'];
  const apiKey = config.apiKey || process.env['GEMINI_API_KEY'];
  if (!apiKey) {
    throw new Error(
      'GEMINI_API_KEY is required when GEMINI_CLI_API_PROVIDER=anthropic.',
    );
  }

  const version = await getVersion();
  const model = gcConfig.getModel();
  const customHeadersEnv =
    process.env['GEMINI_CLI_CUSTOM_HEADERS'] || undefined;
  const customHeadersMap = parseCustomHeaders(customHeadersEnv);

  const anthropicGenerator = new AnthropicContentGenerator({
    apiKey,
    baseURL,
    defaultHeaders: {
      ...customHeadersMap,
      'User-Agent': `GeminiCLI/${version}/${model} (${process.platform}; ${process.arch})`,
    },
  });

  return new LoggingContentGenerator(anthropicGenerator, gcConfig);
}
