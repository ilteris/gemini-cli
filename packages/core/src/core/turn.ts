/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  createUserContent,
  type PartListUnion,
  type GenerateContentResponse,
  type FunctionCall,
  type FunctionDeclaration,
  type FinishReason,
  type GenerateContentResponseUsageMetadata,
} from '@google/genai';
import type {
  ToolCallConfirmationDetails,
  ToolResult,
} from '../tools/tools.js';
import { getResponseText } from '../utils/partUtils.js';
import { reportError } from '../utils/errorReporting.js';
import { ragLogger, type RagSnippet } from '../utils/ragLogger.js';
import {
  getErrorMessage,
  UnauthorizedError,
  toFriendlyError,
} from '../utils/errors.js';
import { InvalidStreamError, type GeminiChat } from './geminiChat.js';
import { parseThought, type ThoughtSummary } from '../utils/thoughtUtils.js';
import type { ModelConfigKey } from '../services/modelConfigService.js';
import { getCitations } from '../utils/generateContentResponseUtilities.js';
import { LlmRole } from '../telemetry/types.js';
import { populateToolDisplay } from '../agent/tool-display-utils.js';

import {
  type ToolCallRequestInfo,
  type ToolCallResponseInfo,
} from '../scheduler/types.js';

export interface ServerTool {
  name: string;
  schema: FunctionDeclaration;
  // The execute method signature might differ slightly or be wrapped
  execute(
    params: Record<string, unknown>,
    signal?: AbortSignal,
  ): Promise<ToolResult>;
  shouldConfirmExecute(
    params: Record<string, unknown>,
    abortSignal: AbortSignal,
  ): Promise<ToolCallConfirmationDetails | false>;
}

export enum GeminiEventType {
  Content = 'content',
  ToolCallRequest = 'tool_call_request',
  ToolCallResponse = 'tool_call_response',
  ToolCallConfirmation = 'tool_call_confirmation',
  UserCancelled = 'user_cancelled',
  Error = 'error',
  ChatCompressed = 'chat_compressed',
  Thought = 'thought',
  MaxSessionTurns = 'max_session_turns',
  Finished = 'finished',
  LoopDetected = 'loop_detected',
  Citation = 'citation',
  Retry = 'retry',
  ContextWindowWillOverflow = 'context_window_will_overflow',
  InvalidStream = 'invalid_stream',
  ModelInfo = 'model_info',
  AgentExecutionStopped = 'agent_execution_stopped',
  AgentExecutionBlocked = 'agent_execution_blocked',
}

export type ServerGeminiRetryEvent = {
  type: GeminiEventType.Retry;
};

export type ServerGeminiAgentExecutionStoppedEvent = {
  type: GeminiEventType.AgentExecutionStopped;
  value: {
    reason: string;
    systemMessage?: string;
    contextCleared?: boolean;
  };
};

export type ServerGeminiAgentExecutionBlockedEvent = {
  type: GeminiEventType.AgentExecutionBlocked;
  value: {
    reason: string;
    systemMessage?: string;
    contextCleared?: boolean;
  };
};

export type ServerGeminiContextWindowWillOverflowEvent = {
  type: GeminiEventType.ContextWindowWillOverflow;
  value: {
    estimatedRequestTokenCount: number;
    remainingTokenCount: number;
  };
};

export type ServerGeminiInvalidStreamEvent = {
  type: GeminiEventType.InvalidStream;
};

export type ServerGeminiModelInfoEvent = {
  type: GeminiEventType.ModelInfo;
  value: string;
};

export interface StructuredError {
  message: string;
  status?: number;
}

export interface GeminiErrorEventValue {
  error: unknown;
}

export interface GeminiFinishedEventValue {
  reason: FinishReason | undefined;
  usageMetadata: GenerateContentResponseUsageMetadata | undefined;
}

export interface ServerToolCallConfirmationDetails {
  request: ToolCallRequestInfo;
  details: ToolCallConfirmationDetails;
}

export type ServerGeminiContentEvent = {
  type: GeminiEventType.Content;
  value: string;
  traceId?: string;
};

export type ServerGeminiThoughtEvent = {
  type: GeminiEventType.Thought;
  value: ThoughtSummary;
  traceId?: string;
};

export type ServerGeminiToolCallRequestEvent = {
  type: GeminiEventType.ToolCallRequest;
  value: ToolCallRequestInfo;
};

export type ServerGeminiToolCallResponseEvent = {
  type: GeminiEventType.ToolCallResponse;
  value: ToolCallResponseInfo;
};

export type ServerGeminiToolCallConfirmationEvent = {
  type: GeminiEventType.ToolCallConfirmation;
  value: ServerToolCallConfirmationDetails;
};

export type ServerGeminiUserCancelledEvent = {
  type: GeminiEventType.UserCancelled;
};

export type ServerGeminiErrorEvent = {
  type: GeminiEventType.Error;
  value: GeminiErrorEventValue;
};

export enum CompressionStatus {
  /** The compression was successful */
  COMPRESSED = 1,

  /** The compression failed due to the compression inflating the token count */
  COMPRESSION_FAILED_INFLATED_TOKEN_COUNT,

  /** The compression failed due to an error counting tokens */
  COMPRESSION_FAILED_TOKEN_COUNT_ERROR,

  /** The compression failed because the summary was empty */
  COMPRESSION_FAILED_EMPTY_SUMMARY,

  /** The compression was not necessary and no action was taken */
  NOOP,

  /** The compression was skipped due to previous failure, but content was truncated to budget */
  CONTENT_TRUNCATED,
}

export interface ChatCompressionInfo {
  originalTokenCount: number;
  newTokenCount: number;
  compressionStatus: CompressionStatus;
}

export type ServerGeminiChatCompressedEvent = {
  type: GeminiEventType.ChatCompressed;
  value: ChatCompressionInfo | null;
};

export type ServerGeminiMaxSessionTurnsEvent = {
  type: GeminiEventType.MaxSessionTurns;
};

export type ServerGeminiFinishedEvent = {
  type: GeminiEventType.Finished;
  value: GeminiFinishedEventValue;
};

export type ServerGeminiLoopDetectedEvent = {
  type: GeminiEventType.LoopDetected;
};

export type ServerGeminiCitationEvent = {
  type: GeminiEventType.Citation;
  value: string;
};

// The original union type, now composed of the individual types
export type ServerGeminiStreamEvent =
  | ServerGeminiChatCompressedEvent
  | ServerGeminiCitationEvent
  | ServerGeminiContentEvent
  | ServerGeminiErrorEvent
  | ServerGeminiFinishedEvent
  | ServerGeminiLoopDetectedEvent
  | ServerGeminiMaxSessionTurnsEvent
  | ServerGeminiThoughtEvent
  | ServerGeminiToolCallConfirmationEvent
  | ServerGeminiToolCallRequestEvent
  | ServerGeminiToolCallResponseEvent
  | ServerGeminiUserCancelledEvent
  | ServerGeminiRetryEvent
  | ServerGeminiContextWindowWillOverflowEvent
  | ServerGeminiInvalidStreamEvent
  | ServerGeminiModelInfoEvent
  | ServerGeminiAgentExecutionStoppedEvent
  | ServerGeminiAgentExecutionBlockedEvent;

// A turn manages the agentic loop turn within the server context.
export class Turn {
  private callCounter = 0;

  readonly pendingToolCalls: ToolCallRequestInfo[] = [];
  private debugResponses: GenerateContentResponse[] = [];
  private pendingCitations = new Set<string>();
  private cachedResponseText: string | undefined = undefined;
  finishReason: FinishReason | undefined = undefined;
  private hasLoggedRagTrace = false;

  constructor(
    private readonly chat: GeminiChat,
    private readonly prompt_id: string,
  ) {}

  // The run method yields simpler events suitable for server logic
  async *run(
    modelConfigKey: ModelConfigKey,
    req: PartListUnion,
    signal: AbortSignal,
    displayContent?: PartListUnion,
    role: LlmRole = LlmRole.MAIN,
  ): AsyncGenerator<ServerGeminiStreamEvent> {
    try {
      // Note: This assumes `sendMessageStream` yields events like
      // { type: StreamEventType.RETRY } or { type: StreamEventType.CHUNK, value: GenerateContentResponse }
      const responseStream = await this.chat.sendMessageStream(
        modelConfigKey,
        req,
        this.prompt_id,
        signal,
        role,
        displayContent,
      );

      for await (const streamEvent of responseStream) {
        if (signal?.aborted) {
          yield { type: GeminiEventType.UserCancelled };
          return;
        }

        // Handle the new RETRY event
        if (streamEvent.type === 'retry') {
          yield { type: GeminiEventType.Retry };
          continue; // Skip to the next event in the stream
        }

        if (streamEvent.type === 'agent_execution_stopped') {
          yield {
            type: GeminiEventType.AgentExecutionStopped,
            value: { reason: streamEvent.reason },
          };
          return;
        }

        if (streamEvent.type === 'agent_execution_blocked') {
          yield {
            type: GeminiEventType.AgentExecutionBlocked,
            value: { reason: streamEvent.reason },
          };
          continue;
        }

        // Assuming other events are chunks with a `value` property
        const resp = streamEvent.value;
        if (!resp) continue; // Skip if there's no response body

        // Log RAG trace if enabled (only once per turn to avoid log bloat on streams)
        if (
          !this.hasLoggedRagTrace &&
          this.chat.context.config.getLogRagSnippets?.()
        ) {
          let ragStatus: string | undefined;
          let snippets: RagSnippet[] | undefined;

          if (
            typeof resp === 'object' &&
            resp !== null &&
            'metadata' in resp &&
            typeof resp.metadata === 'object' &&
            resp.metadata !== null
          ) {
            const metadata = resp.metadata as {
              ragStatus?: string;
              snippets?: RagSnippet[];
            };
            ragStatus = metadata.ragStatus;
            snippets = metadata.snippets;
          }

          if (ragStatus || snippets) {
            ragLogger.log({
              sessionId: this.chat.context.config.getSessionId(),
              ragStatus: ragStatus ?? 'UNKNOWN',
              snippets: snippets ?? [],
            });
            this.hasLoggedRagTrace = true;
          }
        }

        this.debugResponses.push(resp);

        const traceId = resp.responseId;

        const parts = resp.candidates?.[0]?.content?.parts ?? [];
        for (const part of parts) {
          if (part.thought) {
            const thought = parseThought(part.text ?? '');
            yield {
              type: GeminiEventType.Thought,
              value: thought,
              traceId,
            };
          }
        }

        const text = getResponseText(resp);
        if (text) {
          yield { type: GeminiEventType.Content, value: text, traceId };
        }

        // Handle function calls (requesting tool execution)
        const functionCalls = resp.functionCalls ?? [];
        for (const fnCall of functionCalls) {
          const event = this.handlePendingFunctionCall(fnCall, traceId);
          if (event) {
            yield event;
          }
        }

        for (const citation of getCitations(resp)) {
          this.pendingCitations.add(citation);
        }

        // Check if response was truncated or stopped for various reasons
        const finishReason = resp.candidates?.[0]?.finishReason;

        // This is the key change: Only yield 'Finished' if there is a finishReason.
        if (finishReason) {
          if (this.pendingCitations.size > 0) {
            yield {
              type: GeminiEventType.Citation,
              value: `Citations:\n${[...this.pendingCitations].sort().join('\n')}`,
            };
            this.pendingCitations.clear();
          }

          this.finishReason = finishReason;
          yield {
            type: GeminiEventType.Finished,
            value: {
              reason: finishReason,
              usageMetadata: resp.usageMetadata,
            },
          };
        }
      }
    } catch (e) {
      if (signal.aborted) {
        yield { type: GeminiEventType.UserCancelled };
        // Regular cancellation error, fail gracefully.
        return;
      }

      if (e instanceof InvalidStreamError) {
        yield { type: GeminiEventType.InvalidStream };
        return;
      }

      const error = toFriendlyError(e);
      if (error instanceof UnauthorizedError) {
        throw error;
      }

      const contextForReport = [
        ...this.chat.getHistory(/*curated*/ true),
        createUserContent(req),
      ];
      await reportError(
        error,
        'Error when talking to Gemini API',
        contextForReport,
        'Turn.run-sendMessageStream',
      );
      const status =
        typeof error === 'object' &&
        error !== null &&
        'status' in error &&
        typeof (error as { status: unknown }).status === 'number'
          ? // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
            (error as { status: number }).status
          : undefined;
      const structuredError: StructuredError = {
        message: getErrorMessage(error),
        status,
      };
      await this.chat.maybeIncludeSchemaDepthContext(structuredError);
      yield { type: GeminiEventType.Error, value: { error: structuredError } };
      return;
    }
  }

  private handlePendingFunctionCall(
    fnCall: FunctionCall,
    traceId?: string,
  ): ServerGeminiStreamEvent | null {
    const name = fnCall.name?.trim() || 'generic_tool';
    // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
    const args = (fnCall.args as Record<string, unknown>) || {};
    const rawCallId =
      fnCall.id ??
      (this.chat.context.config.isContextManagementEnabled()
        ? `synth_${this.prompt_id}_${Date.now()}_${this.callCounter++}`
        : `${name}_${Date.now()}_${this.callCounter++}`);

    const callId = rawCallId.startsWith(`${name}__`)
      ? rawCallId
      : `${name}__${rawCallId}`;

    // Mutate the function call object ID so that history consolidation inherits it
    fnCall.id = callId;

    const tool = this.chat.loopContext.toolRegistry.getTool(name);
    let display;
    if (tool) {
      let invocation;
      try {
        invocation = tool.build(args);
      } catch {
        // Ignore build errors for request display purposes
      }
      display = populateToolDisplay({
        name,
        invocation,
        displayName: tool.displayName,
      });

      // Fallback to static description if invocation failed or didn't provide one
      if (!display.description) {
        display.description = tool.description;
      }
    }

    const toolCallRequest: ToolCallRequestInfo = {
      callId,
      name,
      args,
      display,
      isClientInitiated: false,
      prompt_id: this.prompt_id,
      traceId,
    };

    this.pendingToolCalls.push(toolCallRequest);

    // Yield a request for the tool call, not the pending/confirming status
    return { type: GeminiEventType.ToolCallRequest, value: toolCallRequest };
  }

  getDebugResponses(): GenerateContentResponse[] {
    return this.debugResponses;
  }

  /**
   * Get the concatenated response text from all responses in this turn.
   * This extracts and joins all text content from the model's responses.
   * The result is cached since this is called multiple times per turn.
   *
   * Gemini's stream usually emits pure deltas, but around thought,
   * function-call, or retry boundaries it can re-emit text that already
   * appeared in a previous chunk. A naive `.join(' ')` then writes the
   * overlapping suffix twice. We walk the chunks and, before appending
   * the next one, collapse the longest suffix-of-accumulated that
   * prefixes-the-next (>= MIN_OVERLAP, capped at MAX_OVERLAP_SCAN) so
   * the accumulated text matches what the model actually produced. The
   * short MIN_OVERLAP guard keeps legitimate small repeats ("the the",
   * closing braces in code) intact; the MAX_OVERLAP_SCAN cap keeps the
   * suffix search linear-ish for large code-gen chunks.
   * See SOUL-SOUL_DESKTOP-264.
   */
  getResponseText(): string {
    if (this.cachedResponseText === undefined) {
      const texts = this.debugResponses
        .map((response) => getResponseText(response))
        .filter((text): text is string => text !== null && text.length > 0);
      this.cachedResponseText = collapseStreamOverlap(texts);
    }
    return this.cachedResponseText;
  }
}

/** Minimum overlap (in chars) before we treat a suffix==prefix match as a
 *  re-emission rather than a coincidental short repeat. */
const MIN_OVERLAP = 16;
/** Cap on the suffix length we scan for overlap. Keeps the O(k) tail-check
 *  bounded on multi-KB code-gen chunks. */
const MAX_OVERLAP_SCAN = 4096;

/**
 * Concatenates streamed chunk texts while collapsing self-overlaps where the
 * head of `next` reappears the tail of the accumulated text. Used by
 * Turn.getResponseText() — see comment there for context.
 *
 * Algorithm: for each subsequent chunk, find the largest k in
 * [MIN_OVERLAP, min(|accum|, |next|, MAX_OVERLAP_SCAN)] such that
 * accum.endsWith(next.slice(0, k)). Append next.slice(k); if no overlap
 * found, fall back to a single-space separator to preserve the prior
 * `.join(' ')` semantics for true deltas.
 */
export function collapseStreamOverlap(texts: string[]): string {
  let result = '';
  for (const text of texts) {
    if (!result) {
      result = text;
      continue;
    }
    const maxK = Math.min(result.length, text.length, MAX_OVERLAP_SCAN);
    let overlap = 0;
    for (let k = maxK; k >= MIN_OVERLAP; k--) {
      // endsWith with a position arg would avoid the slice allocation, but
      // the slice form is clearer and the cap keeps the cost bounded.
      if (result.endsWith(text.slice(0, k))) {
        overlap = k;
        break;
      }
    }
    result += overlap > 0 ? text.slice(overlap) : ' ' + text;
  }
  return result;
}
