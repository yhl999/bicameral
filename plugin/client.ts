export interface GraphitiFact {
  uuid?: string;
  name?: string;
  fact: string;
}

export interface GraphitiEntity {
  uuid?: string;
  name?: string;
  summary: string;
}

export interface GraphitiSearchResults {
  facts: GraphitiFact[];
  entities?: GraphitiEntity[];
}

export interface GraphitiMessage {
  content: string;
  role_type: 'user' | 'assistant' | 'system';
  role?: string | null;
  name?: string;
  uuid?: string | null;
  timestamp?: string;
  source_description?: string;
}

export interface GraphitiClientConfig {
  baseUrl: string;
  apiKey?: string;
  recallTimeoutMs: number;
  captureTimeoutMs: number;
  maxFacts: number;
}

interface JsonRpcError {
  code: number;
  message: string;
  data?: unknown;
}

interface JsonRpcEnvelope {
  jsonrpc?: string;
  id?: string | number | null;
  method?: string;
  params?: Record<string, unknown>;
  result?: unknown;
  error?: JsonRpcError;
}

interface McpToolTextContent {
  type?: string;
  text?: string;
}

interface McpToolResultEnvelope {
  structuredContent?: unknown;
  content?: McpToolTextContent[];
  isError?: boolean;
}

const MCP_SESSION_HEADER = 'mcp-session-id';

const normalizeBaseUrl = (baseUrl: string): string => baseUrl.replace(/\/$/, '');

/**
 * Resolve streamable-HTTP MCP endpoint from operator config.
 *
 * Backwards compatibility:
 * - Existing deployments often set graphitiBaseUrl=http://localhost:8000
 * - The MCP server exposes transport on /mcp (or /mcp/)
 *
 * So if /mcp is missing, append it automatically.
 */
const resolveMcpUrl = (baseUrl: string): string => {
  const normalized = normalizeBaseUrl(baseUrl);
  if (/\/mcp$/i.test(normalized)) {
    return `${normalized}/`;
  }
  if (/\/mcp\/$/i.test(baseUrl)) {
    return baseUrl;
  }
  return `${normalized}/mcp/`;
};

const asText = (value: unknown): string => {
  if (typeof value === 'string') {
    return value;
  }
  if (value == null) {
    return '';
  }
  return String(value);
};

const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null;

const parseMaybeJson = (text: string): unknown => {
  const trimmed = text.trim();
  if (!trimmed) {
    return {};
  }
  return JSON.parse(trimmed) as unknown;
};

const decodeBody = (contentType: string, bodyText: string): unknown => {
  const type = (contentType || '').toLowerCase();

  if (type.startsWith('text/event-stream')) {
    const dataLines = bodyText
      .split(/\r?\n/)
      .filter((line) => line.startsWith('data:'))
      .map((line) => line.slice('data:'.length).trim())
      .filter((line) => line.length > 0);

    for (let index = dataLines.length - 1; index >= 0; index -= 1) {
      const candidate = dataLines[index];
      try {
        return JSON.parse(candidate) as unknown;
      } catch {
        // Keep scanning older SSE data lines.
      }
    }
    return {};
  }

  try {
    return parseMaybeJson(bodyText);
  } catch {
    return {
      raw: bodyText,
    };
  }
};

export class GraphitiClient {
  private readonly baseUrl: string;
  private readonly mcpUrl: string;
  private readonly apiKey?: string;
  private readonly recallTimeoutMs: number;
  private readonly captureTimeoutMs: number;
  private readonly maxFacts: number;

  private mcpSessionId?: string;
  private initializePromise?: Promise<void>;
  private requestId = 0;

  constructor(config: GraphitiClientConfig) {
    this.baseUrl = normalizeBaseUrl(config.baseUrl);
    this.mcpUrl = resolveMcpUrl(config.baseUrl);
    this.apiKey = config.apiKey;
    this.recallTimeoutMs = config.recallTimeoutMs;
    this.captureTimeoutMs = config.captureTimeoutMs;
    this.maxFacts = config.maxFacts;
  }

  async search(query: string, groupIds?: string[]): Promise<GraphitiSearchResults> {
    const toolResult = await this.callTool(
      'search_memory_facts',
      {
        query,
        group_ids: groupIds && groupIds.length > 0 ? groupIds : undefined,
        max_facts: this.maxFacts,
      },
      this.recallTimeoutMs,
    );

    const payload = this.extractToolPayload(toolResult);
    const rawFacts = Array.isArray(payload.facts) ? payload.facts : [];

    const facts: GraphitiFact[] = rawFacts
      .map((item) => {
        if (!isObject(item)) {
          return null;
        }

        const factText = asText(item.fact).trim();
        if (!factText) {
          return null;
        }

        const fact: GraphitiFact = { fact: factText };
        const uuid = asText(item.uuid).trim();
        const name = asText(item.name).trim();
        if (uuid) {
          fact.uuid = uuid;
        }
        if (name) {
          fact.name = name;
        }
        return fact;
      })
      .filter((item): item is GraphitiFact => Boolean(item));

    return {
      facts,
      entities: [],
    };
  }

  async ingestMessages(groupId: string, messages: GraphitiMessage[]): Promise<void> {
    if (!groupId || messages.length === 0) {
      return;
    }

    const episodeBody = messages
      .map((message) => `${message.role_type}: ${message.content}`)
      .join('\n');

    await this.callTool(
      'add_memory',
      {
        name: `openclaw_turn_${Date.now()}`,
        episode_body: episodeBody,
        group_id: groupId,
        source: 'message',
        source_description: 'OpenClaw bicameral plugin capture',
      },
      this.captureTimeoutMs,
    );
  }

  private async ensureInitialized(timeoutMs: number): Promise<void> {
    if (!this.initializePromise) {
      this.initializePromise = (async () => {
        await this.rpcCall(
          'initialize',
          {
            protocolVersion: '2024-11-05',
            capabilities: {},
            clientInfo: {
              name: 'openclaw-bicameral-plugin',
              version: '0.1.0',
            },
          },
          timeoutMs,
        );

        // Notification: no response body is required/expected.
        await this.rpcNotify('notifications/initialized', {}, timeoutMs);
      })().catch((error) => {
        this.initializePromise = undefined;
        throw error;
      });
    }

    await this.initializePromise;
  }

  private async callTool(
    name: string,
    argumentsPayload: Record<string, unknown>,
    timeoutMs: number,
  ): Promise<unknown> {
    await this.ensureInitialized(timeoutMs);

    return this.rpcCall(
      'tools/call',
      {
        name,
        arguments: argumentsPayload,
      },
      timeoutMs,
    );
  }

  private async rpcNotify(
    method: string,
    params: Record<string, unknown> | undefined,
    timeoutMs: number,
  ): Promise<void> {
    const payload: JsonRpcEnvelope = {
      jsonrpc: '2.0',
      method,
      ...(params ? { params } : {}),
    };
    await this.postAndDecode(payload, timeoutMs, true);
  }

  private async rpcCall(
    method: string,
    params: Record<string, unknown> | undefined,
    timeoutMs: number,
  ): Promise<unknown> {
    const payload: JsonRpcEnvelope = {
      jsonrpc: '2.0',
      id: ++this.requestId,
      method,
      ...(params ? { params } : {}),
    };

    let result = await this.postAndDecode(payload, timeoutMs, true);

    // FastMCP can return a 400 with a fresh session header before first
    // successful request. If that happens, retry once with the new header.
    if (
      result.status === 400
      && result.bodyText.includes('Missing session ID')
      && !result.hadSessionHeader
      && Boolean(this.mcpSessionId)
    ) {
      result = await this.postAndDecode(payload, timeoutMs, true);
    }

    if (result.status >= 400) {
      if (isObject(result.decoded) && isObject((result.decoded as { error?: unknown }).error)) {
        const rpcError = (result.decoded as { error: JsonRpcError }).error;
        throw new Error(`Graphiti MCP error ${rpcError.code}: ${rpcError.message}`);
      }
      throw new Error(`Graphiti MCP API error ${result.status}`);
    }

    if (!isObject(result.decoded)) {
      throw new Error('Graphiti MCP malformed response');
    }

    const envelope = result.decoded as JsonRpcEnvelope;
    if (envelope.error) {
      throw new Error(`Graphiti MCP error ${envelope.error.code}: ${envelope.error.message}`);
    }

    return envelope.result;
  }

  private async postAndDecode(
    payload: JsonRpcEnvelope,
    timeoutMs: number,
    includeSessionHeader: boolean,
  ): Promise<{
    status: number;
    bodyText: string;
    decoded: unknown;
    hadSessionHeader: boolean;
  }> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs);

    try {
      const headers: Record<string, string> = {
        Accept: 'application/json, text/event-stream',
        'Content-Type': 'application/json',
      };

      if (this.apiKey) {
        headers.Authorization = `Bearer ${this.apiKey}`;
      }

      const hadSessionHeader = includeSessionHeader && Boolean(this.mcpSessionId);
      if (includeSessionHeader && this.mcpSessionId) {
        headers[MCP_SESSION_HEADER] = this.mcpSessionId;
      }

      const response = await fetch(this.mcpUrl, {
        method: 'POST',
        headers,
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      const sessionId = response.headers.get(MCP_SESSION_HEADER);
      if (sessionId && !this.mcpSessionId) {
        this.mcpSessionId = sessionId;
      }

      const bodyText = await response.text();
      const decoded = decodeBody(response.headers.get('content-type') ?? '', bodyText);

      return {
        status: response.status,
        bodyText,
        decoded,
        hadSessionHeader,
      };
    } finally {
      clearTimeout(timeout);
    }
  }

  private extractToolPayload(toolResult: unknown): Record<string, unknown> {
    if (!isObject(toolResult)) {
      return {};
    }

    const envelope = toolResult as McpToolResultEnvelope;
    if (isObject(envelope.structuredContent)) {
      return envelope.structuredContent;
    }

    const content = Array.isArray(envelope.content) ? envelope.content : [];
    const textBlocks = content
      .map((item) => (isObject(item) ? asText(item.text) : ''))
      .filter((text) => text.trim().length > 0);

    if (envelope.isError) {
      const reason = textBlocks.length > 0 ? textBlocks.join('\n') : 'unknown error';
      throw new Error(`Graphiti MCP tool error: ${reason}`);
    }

    if (textBlocks.length > 0) {
      try {
        const parsed = parseMaybeJson(textBlocks[textBlocks.length - 1]);
        if (isObject(parsed)) {
          return parsed;
        }
      } catch {
        // Fall through to object checks below.
      }
    }

    if (Array.isArray((toolResult as { facts?: unknown[] }).facts)) {
      return toolResult as Record<string, unknown>;
    }

    return {};
  }
}
