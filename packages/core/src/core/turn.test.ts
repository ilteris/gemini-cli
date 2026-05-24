/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  Turn,
  GeminiEventType,
  collapseStreamOverlap,
  type ServerGeminiToolCallRequestEvent,
  type ServerGeminiErrorEvent,
} from './turn.js';
import type { GenerateContentResponse, Part, Content } from '@google/genai';
import { reportError } from '../utils/errorReporting.js';
import {
  InvalidStreamError,
  StreamEventType,
  type GeminiChat,
} from './geminiChat.js';
import { LlmRole } from '../telemetry/types.js';

const mockSendMessageStream = vi.fn();
const mockGetHistory = vi.fn();
const mockMaybeIncludeSchemaDepthContext = vi.fn();

vi.mock('@google/genai', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@google/genai')>();
  const MockChat = vi.fn().mockImplementation(() => ({
    sendMessageStream: mockSendMessageStream,
    getHistory: mockGetHistory,
    maybeIncludeSchemaDepthContext: mockMaybeIncludeSchemaDepthContext,
  }));
  return {
    ...actual,
    Chat: MockChat,
  };
});

vi.mock('../utils/errorReporting', () => ({
  reportError: vi.fn(),
}));

describe('Turn', () => {
  let turn: Turn;
  // Define a type for the mocked Chat instance for clarity
  type MockedChatInstance = {
    sendMessageStream: typeof mockSendMessageStream;
    getHistory: typeof mockGetHistory;
    maybeIncludeSchemaDepthContext: typeof mockMaybeIncludeSchemaDepthContext;
    context: { config: { isContextManagementEnabled: () => boolean } };
    loopContext?: {
      toolRegistry: {
        getTool: (name: string) => unknown;
      };
    };
  };
  let mockChatInstance: MockedChatInstance;

  beforeEach(() => {
    vi.resetAllMocks();
    mockChatInstance = {
      sendMessageStream: mockSendMessageStream,
      getHistory: mockGetHistory,
      maybeIncludeSchemaDepthContext: mockMaybeIncludeSchemaDepthContext,
      context: {
        config: {
          isContextManagementEnabled: () => false,
        },
      },
      loopContext: {
        toolRegistry: {
          getTool: vi.fn().mockReturnValue(undefined),
        },
      },
    };
    turn = new Turn(mockChatInstance as unknown as GeminiChat, 'prompt-id-1');
    mockGetHistory.mockReturnValue([]);
    mockSendMessageStream.mockResolvedValue((async function* () {})());
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('constructor', () => {
    it('should initialize pendingToolCalls and debugResponses', () => {
      expect(turn.pendingToolCalls).toEqual([]);
      expect(turn.getDebugResponses()).toEqual([]);
    });
  });

  describe('run', () => {
    it('should yield content events for text parts', async () => {
      const mockResponseStream = (async function* () {
        yield {
          type: StreamEventType.CHUNK,
          value: {
            candidates: [{ content: { parts: [{ text: 'Hello' }] } }],
          } as GenerateContentResponse,
        };
        yield {
          type: StreamEventType.CHUNK,
          value: {
            candidates: [{ content: { parts: [{ text: ' world' }] } }],
          } as GenerateContentResponse,
        };
      })();
      mockSendMessageStream.mockResolvedValue(mockResponseStream);

      const events = [];
      const reqParts: Part[] = [{ text: 'Hi' }];
      for await (const event of turn.run(
        { model: 'gemini' },
        reqParts,
        new AbortController().signal,
      )) {
        events.push(event);
      }

      expect(mockSendMessageStream).toHaveBeenCalledWith(
        { model: 'gemini' },
        reqParts,
        'prompt-id-1',
        expect.any(AbortSignal),
        LlmRole.MAIN,
        undefined,
      );

      expect(events).toEqual([
        { type: GeminiEventType.Content, value: 'Hello' },
        { type: GeminiEventType.Content, value: ' world' },
      ]);
      expect(turn.getDebugResponses().length).toBe(2);
    });

    it('should yield tool_call_request events for function calls', async () => {
      const mockResponseStream = (async function* () {
        yield {
          type: StreamEventType.CHUNK,
          value: {
            functionCalls: [
              {
                id: 'fc1',
                name: 'tool1',
                args: { arg1: 'val1' },
                isClientInitiated: false,
              },
              {
                name: 'tool2',
                args: { arg2: 'val2' },
                isClientInitiated: false,
              }, // No ID
            ],
          } as unknown as GenerateContentResponse,
        };
      })();
      mockSendMessageStream.mockResolvedValue(mockResponseStream);

      const events = [];
      const reqParts: Part[] = [{ text: 'Use tools' }];
      for await (const event of turn.run(
        { model: 'gemini' },
        reqParts,
        new AbortController().signal,
      )) {
        events.push(event);
      }

      expect(events.length).toBe(2);
      const event1 = events[0] as ServerGeminiToolCallRequestEvent;
      expect(event1.type).toBe(GeminiEventType.ToolCallRequest);
      expect(event1.value).toEqual(
        expect.objectContaining({
          callId: 'tool1__fc1',
          name: 'tool1',
          args: { arg1: 'val1' },
          isClientInitiated: false,
        }),
      );
      expect(turn.pendingToolCalls[0]).toEqual(event1.value);

      const event2 = events[1] as ServerGeminiToolCallRequestEvent;
      expect(event2.type).toBe(GeminiEventType.ToolCallRequest);
      expect(event2.value).toEqual(
        expect.objectContaining({
          name: 'tool2',
          args: { arg2: 'val2' },
          isClientInitiated: false,
        }),
      );
      expect(event2.value.callId).toEqual(
        expect.stringMatching(/^tool2__tool2_\d{13}_\d+$/),
      );
      expect(turn.pendingToolCalls[1]).toEqual(event2.value);
      expect(turn.getDebugResponses().length).toBe(1);
    });

    it('should yield UserCancelled event if signal is aborted', async () => {
      const abortController = new AbortController();
      const mockResponseStream = (async function* () {
        yield {
          type: StreamEventType.CHUNK,
          value: {
            candidates: [{ content: { parts: [{ text: 'First part' }] } }],
          } as GenerateContentResponse,
        };
        abortController.abort();
        yield {
          type: StreamEventType.CHUNK,
          value: {
            candidates: [
              {
                content: {
                  parts: [{ text: 'Second part - should not be processed' }],
                },
              },
            ],
          } as GenerateContentResponse,
        };
      })();
      mockSendMessageStream.mockResolvedValue(mockResponseStream);

      const events = [];
      const reqParts: Part[] = [{ text: 'Test abort' }];
      for await (const event of turn.run(
        { model: 'gemini' },
        reqParts,
        abortController.signal,
      )) {
        events.push(event);
      }
      expect(events).toEqual([
        { type: GeminiEventType.Content, value: 'First part' },
        { type: GeminiEventType.UserCancelled },
      ]);
      expect(turn.getDebugResponses().length).toBe(1);
    });

    it('should yield InvalidStream event if sendMessageStream throws InvalidStreamError', async () => {
      const error = new InvalidStreamError(
        'Test invalid stream',
        'NO_FINISH_REASON',
      );
      mockSendMessageStream.mockRejectedValue(error);
      const reqParts: Part[] = [{ text: 'Trigger invalid stream' }];

      const events = [];
      for await (const event of turn.run(
        { model: 'gemini' },
        reqParts,
        new AbortController().signal,
      )) {
        events.push(event);
      }

      expect(events).toEqual([{ type: GeminiEventType.InvalidStream }]);
      expect(turn.getDebugResponses().length).toBe(0);
      expect(reportError).not.toHaveBeenCalled(); // Should not report as error
    });

    it('should yield Error event and report if sendMessageStream throws', async () => {
      const error = new Error('API Error');
      mockSendMessageStream.mockRejectedValue(error);
      const reqParts: Part[] = [{ text: 'Trigger error' }];
      const historyContent: Content[] = [
        { role: 'model', parts: [{ text: 'Previous history' }] },
      ];
      mockGetHistory.mockReturnValue(historyContent);
      mockMaybeIncludeSchemaDepthContext.mockResolvedValue(undefined);
      const events = [];
      for await (const event of turn.run(
        { model: 'gemini' },
        reqParts,
        new AbortController().signal,
      )) {
        events.push(event);
      }

      expect(events.length).toBe(1);
      const errorEvent = events[0] as ServerGeminiErrorEvent;
      expect(errorEvent.type).toBe(GeminiEventType.Error);
      expect(errorEvent.value).toEqual({
        error: {
          message: 'API Error',
          status: undefined,
        },
      });
      expect(turn.getDebugResponses().length).toBe(0);
      expect(reportError).toHaveBeenCalledWith(
        error,
        'Error when talking to Gemini API',
        [...historyContent, { role: 'user', parts: reqParts }],
        'Turn.run-sendMessageStream',
      );
    });

    it('should handle function calls with undefined name or args', async () => {
      const mockResponseStream = (async function* () {
        yield {
          type: StreamEventType.CHUNK,
          value: {
            candidates: [],
            functionCalls: [
              // Add `id` back to the mock to match what the code expects
              { id: 'fc1', name: undefined, args: { arg1: 'val1' } },
              { id: 'fc2', name: 'tool2', args: undefined },
              { id: 'fc3', name: undefined, args: undefined },
            ],
          },
        };
      })();
      mockSendMessageStream.mockResolvedValue(mockResponseStream);

      const events = [];
      for await (const event of turn.run(
        { model: 'gemini' },
        [{ text: 'Test undefined tool parts' }],
        new AbortController().signal,
      )) {
        events.push(event);
      }

      expect(events.length).toBe(3);

      // Assertions for each specific tool call event
      const event1 = events[0] as ServerGeminiToolCallRequestEvent;
      expect(event1.value).toMatchObject({
        callId: 'generic_tool__fc1',
        name: 'generic_tool',
        args: { arg1: 'val1' },
      });

      const event2 = events[1] as ServerGeminiToolCallRequestEvent;
      expect(event2.value).toMatchObject({
        callId: 'tool2__fc2',
        name: 'tool2',
        args: {},
      });

      const event3 = events[2] as ServerGeminiToolCallRequestEvent;
      expect(event3.value).toMatchObject({
        callId: 'generic_tool__fc3',
        name: 'generic_tool',
        args: {},
      });
    });

    it.each([
      {
        description:
          'should yield finished event when response has finish reason',
        contentText: 'Partial response',
        finishReason: 'STOP',
        usageMetadata: {
          promptTokenCount: 17,
          candidatesTokenCount: 50,
          cachedContentTokenCount: 10,
          thoughtsTokenCount: 5,
          toolUsePromptTokenCount: 2,
        },
      },
      {
        description: 'should yield finished event for MAX_TOKENS finish reason',
        contentText: 'This is a long response that was cut off...',
        finishReason: 'MAX_TOKENS',
        usageMetadata: undefined,
      },
      {
        description: 'should yield finished event for SAFETY finish reason',
        contentText: 'Content blocked',
        finishReason: 'SAFETY',
        usageMetadata: undefined,
      },
    ])('$description', async ({ contentText, finishReason, usageMetadata }) => {
      const mockResponseStream = (async function* () {
        yield {
          type: StreamEventType.CHUNK,
          value: {
            candidates: [
              {
                content: { parts: [{ text: contentText }] },
                finishReason,
              },
            ],
            usageMetadata,
          } as GenerateContentResponse,
        };
      })();
      mockSendMessageStream.mockResolvedValue(mockResponseStream);

      const events = [];
      for await (const event of turn.run(
        { model: 'gemini' },
        [{ text: 'Test' }],
        new AbortController().signal,
      )) {
        events.push(event);
      }

      expect(events).toEqual([
        { type: GeminiEventType.Content, value: contentText },
        {
          type: GeminiEventType.Finished,
          value: { reason: finishReason, usageMetadata },
        },
      ]);
    });

    it('should yield finished event with undefined reason when there is no finish reason', async () => {
      const mockResponseStream = (async function* () {
        yield {
          type: StreamEventType.CHUNK,
          value: {
            candidates: [
              {
                content: {
                  parts: [{ text: 'Response without finish reason' }],
                },
                // No finishReason property
              },
            ],
          },
        };
      })();
      mockSendMessageStream.mockResolvedValue(mockResponseStream);

      const events = [];
      const reqParts: Part[] = [{ text: 'Test no finish reason' }];
      for await (const event of turn.run(
        { model: 'gemini' },
        reqParts,
        new AbortController().signal,
      )) {
        events.push(event);
      }

      expect(events).toEqual([
        {
          type: GeminiEventType.Content,
          value: 'Response without finish reason',
        },
      ]);
    });

    it('should handle multiple responses with different finish reasons', async () => {
      const mockResponseStream = (async function* () {
        yield {
          type: StreamEventType.CHUNK,
          value: {
            candidates: [
              {
                content: { parts: [{ text: 'First part' }] },
                // No finish reason on first response
              },
            ],
          },
        };
        yield {
          value: {
            type: StreamEventType.CHUNK,
            candidates: [
              {
                content: { parts: [{ text: 'Second part' }] },
                finishReason: 'OTHER',
              },
            ],
          },
        };
      })();
      mockSendMessageStream.mockResolvedValue(mockResponseStream);

      const events = [];
      const reqParts: Part[] = [{ text: 'Test multiple responses' }];
      for await (const event of turn.run(
        { model: 'gemini' },
        reqParts,
        new AbortController().signal,
      )) {
        events.push(event);
      }

      expect(events).toEqual([
        { type: GeminiEventType.Content, value: 'First part' },
        { type: GeminiEventType.Content, value: 'Second part' },
        {
          type: GeminiEventType.Finished,
          value: { reason: 'OTHER', usageMetadata: undefined },
        },
      ]);
    });

    it('should yield citation and finished events when response has citationMetadata', async () => {
      const mockResponseStream = (async function* () {
        yield {
          type: StreamEventType.CHUNK,
          value: {
            candidates: [
              {
                content: { parts: [{ text: 'Some text.' }] },
                citationMetadata: {
                  citations: [
                    {
                      uri: 'https://example.com/source1',
                      title: 'Source 1 Title',
                    },
                  ],
                },
                finishReason: 'STOP',
              },
            ],
          },
        };
      })();
      mockSendMessageStream.mockResolvedValue(mockResponseStream);

      const events = [];
      for await (const event of turn.run(
        { model: 'gemini' },
        [{ text: 'Test citations' }],
        new AbortController().signal,
      )) {
        events.push(event);
      }

      expect(events).toEqual([
        { type: GeminiEventType.Content, value: 'Some text.' },
        {
          type: GeminiEventType.Citation,
          value: 'Citations:\n(Source 1 Title) https://example.com/source1',
        },
        {
          type: GeminiEventType.Finished,
          value: { reason: 'STOP', usageMetadata: undefined },
        },
      ]);
    });

    it('should yield a single citation event for multiple citations in one response', async () => {
      const mockResponseStream = (async function* () {
        yield {
          type: StreamEventType.CHUNK,
          value: {
            candidates: [
              {
                content: { parts: [{ text: 'Some text.' }] },
                citationMetadata: {
                  citations: [
                    {
                      uri: 'https://example.com/source2',
                      title: 'Title2',
                    },
                    {
                      uri: 'https://example.com/source1',
                      title: 'Title1',
                    },
                  ],
                },
                finishReason: 'STOP',
              },
            ],
          },
        };
      })();
      mockSendMessageStream.mockResolvedValue(mockResponseStream);

      const events = [];
      for await (const event of turn.run(
        { model: 'gemini' },
        [{ text: 'test' }],
        new AbortController().signal,
      )) {
        events.push(event);
      }

      expect(events).toEqual([
        { type: GeminiEventType.Content, value: 'Some text.' },
        {
          type: GeminiEventType.Citation,
          value:
            'Citations:\n(Title1) https://example.com/source1\n(Title2) https://example.com/source2',
        },
        {
          type: GeminiEventType.Finished,
          value: { reason: 'STOP', usageMetadata: undefined },
        },
      ]);
    });

    it('should not yield citation event if there is no finish reason', async () => {
      const mockResponseStream = (async function* () {
        yield {
          type: StreamEventType.CHUNK,
          value: {
            candidates: [
              {
                content: { parts: [{ text: 'Some text.' }] },
                citationMetadata: {
                  citations: [
                    {
                      uri: 'https://example.com/source1',
                      title: 'Source 1 Title',
                    },
                  ],
                },
                // No finishReason
              },
            ],
          },
        };
      })();
      mockSendMessageStream.mockResolvedValue(mockResponseStream);

      const events = [];
      for await (const event of turn.run(
        { model: 'gemini' },
        [{ text: 'test' }],
        new AbortController().signal,
      )) {
        events.push(event);
      }

      expect(events).toEqual([
        { type: GeminiEventType.Content, value: 'Some text.' },
      ]);
      // No Citation event (but we do get a Finished event with undefined reason)
      expect(events.some((e) => e.type === GeminiEventType.Citation)).toBe(
        false,
      );
    });

    it('should ignore citations without a URI', async () => {
      const mockResponseStream = (async function* () {
        yield {
          type: StreamEventType.CHUNK,
          value: {
            candidates: [
              {
                content: { parts: [{ text: 'Some text.' }] },
                citationMetadata: {
                  citations: [
                    {
                      uri: 'https://example.com/source1',
                      title: 'Good Source',
                    },
                    {
                      // uri is undefined
                      title: 'Bad Source',
                    },
                  ],
                },
                finishReason: 'STOP',
              },
            ],
          },
        };
      })();
      mockSendMessageStream.mockResolvedValue(mockResponseStream);

      const events = [];
      for await (const event of turn.run(
        { model: 'gemini' },
        [{ text: 'test' }],
        new AbortController().signal,
      )) {
        events.push(event);
      }

      expect(events).toEqual([
        { type: GeminiEventType.Content, value: 'Some text.' },
        {
          type: GeminiEventType.Citation,
          value: 'Citations:\n(Good Source) https://example.com/source1',
        },
        {
          type: GeminiEventType.Finished,
          value: { reason: 'STOP', usageMetadata: undefined },
        },
      ]);
    });

    it('should not crash when cancelled request has malformed error', async () => {
      const abortController = new AbortController();

      const errorToThrow = {
        response: {
          data: undefined, // Malformed error data
        },
      };

      mockSendMessageStream.mockImplementation(async () => {
        abortController.abort();
        throw errorToThrow;
      });

      const events = [];
      const reqParts: Part[] = [{ text: 'Test malformed error handling' }];

      for await (const event of turn.run(
        { model: 'gemini' },
        reqParts,
        abortController.signal,
      )) {
        events.push(event);
      }

      expect(events).toEqual([{ type: GeminiEventType.UserCancelled }]);

      expect(reportError).not.toHaveBeenCalled();
    });

    it('should yield a Retry event when it receives one from the chat stream', async () => {
      const mockResponseStream = (async function* () {
        yield { type: StreamEventType.RETRY };
        yield {
          type: StreamEventType.CHUNK,
          value: {
            candidates: [{ content: { parts: [{ text: 'Success' }] } }],
          },
        };
      })();
      mockSendMessageStream.mockResolvedValue(mockResponseStream);

      const events = [];
      for await (const event of turn.run(
        { model: 'gemini' },
        [],
        new AbortController().signal,
      )) {
        events.push(event);
      }

      expect(events).toEqual([
        { type: GeminiEventType.Retry },
        { type: GeminiEventType.Content, value: 'Success' },
      ]);
    });

    it.each([
      {
        description: 'should yield content events with traceId',
        part: { text: 'Hello' },
        responseId: 'trace-123',
        expectedEvent: {
          type: GeminiEventType.Content,
          value: 'Hello',
          traceId: 'trace-123',
        },
      },
      {
        description: 'should yield thought events with traceId',
        part: { text: '[Thought: thinking]', thought: 'thinking' },
        responseId: 'trace-456',
        expectedEvent: {
          type: GeminiEventType.Thought,
          value: { subject: '', description: '[Thought: thinking]' },
          traceId: 'trace-456',
        },
      },
    ])('$description', async ({ part, responseId, expectedEvent }) => {
      const mockResponseStream = (async function* () {
        yield {
          type: StreamEventType.CHUNK,
          value: {
            candidates: [{ content: { parts: [part] } }],
            responseId,
          } as unknown as GenerateContentResponse,
        };
      })();
      mockSendMessageStream.mockResolvedValue(mockResponseStream);

      const events = [];
      for await (const event of turn.run(
        { model: 'gemini' },
        [{ text: 'Hi' }],
        new AbortController().signal,
      )) {
        events.push(event);
      }

      expect(events).toEqual([expectedEvent]);
    });

    it('should process all parts when thought is first part in chunk', async () => {
      const mockResponseStream = (async function* () {
        yield {
          type: StreamEventType.CHUNK,
          value: {
            candidates: [
              {
                content: {
                  parts: [
                    { text: '**Planning** the solution', thought: 'planning' },
                    { text: 'I will help you with that.' },
                  ],
                },
                citationMetadata: {
                  citations: [{ uri: 'https://example.com', title: 'Source' }],
                },
                finishReason: 'STOP',
              },
            ],
            functionCalls: [
              {
                id: 'fc1',
                name: 'ReadFile',
                args: { path: 'file.txt' },
              },
            ],
            responseId: 'trace-789',
          } as unknown as GenerateContentResponse,
        };
      })();
      mockSendMessageStream.mockResolvedValue(mockResponseStream);

      const events = [];
      for await (const event of turn.run(
        { model: 'gemini' },
        [{ text: 'Test mixed content' }],
        new AbortController().signal,
      )) {
        events.push(event);
      }

      // Should yield:
      // 1. Thought event (from first part)
      // 2. Content event (from second part)
      // 3. ToolCallRequest event (from functionCalls)
      // 4. Citation event (from citationMetadata, emitted with finishReason)
      // 5. Finished event (from finishReason)

      expect(events.length).toBe(5);

      const thoughtEvent = events.find(
        (e) => e.type === GeminiEventType.Thought,
      );
      expect(thoughtEvent).toBeDefined();
      expect(thoughtEvent).toMatchObject({
        type: GeminiEventType.Thought,
        value: { subject: 'Planning', description: 'the solution' },
        traceId: 'trace-789',
      });

      const contentEvent = events.find(
        (e) => e.type === GeminiEventType.Content,
      );
      expect(contentEvent).toBeDefined();
      expect(contentEvent).toMatchObject({
        type: GeminiEventType.Content,
        value: 'I will help you with that.',
        traceId: 'trace-789',
      });

      const toolCallEvent = events.find(
        (e) => e.type === GeminiEventType.ToolCallRequest,
      );
      expect(toolCallEvent).toBeDefined();
      expect(toolCallEvent).toMatchObject({
        type: GeminiEventType.ToolCallRequest,
        value: expect.objectContaining({
          callId: 'ReadFile__fc1',
          name: 'ReadFile',
          args: { path: 'file.txt' },
        }),
      });

      const citationEvent = events.find(
        (e) => e.type === GeminiEventType.Citation,
      );
      expect(citationEvent).toBeDefined();
      expect(citationEvent).toMatchObject({
        type: GeminiEventType.Citation,
        value: expect.stringContaining('https://example.com'),
      });

      const finishedEvent = events.find(
        (e) => e.type === GeminiEventType.Finished,
      );
      expect(finishedEvent).toBeDefined();
      expect(finishedEvent).toMatchObject({
        type: GeminiEventType.Finished,
        value: { reason: 'STOP' },
      });
    });
  });

  describe('getDebugResponses', () => {
    it('should return collected debug responses', async () => {
      const resp1 = {
        candidates: [{ content: { parts: [{ text: 'Debug 1' }] } }],
      } as unknown as GenerateContentResponse;
      const resp2 = {
        functionCalls: [{ name: 'debugTool' }],
      } as unknown as GenerateContentResponse;
      const mockResponseStream = (async function* () {
        yield { type: StreamEventType.CHUNK, value: resp1 };
        yield { type: StreamEventType.CHUNK, value: resp2 };
      })();
      mockSendMessageStream.mockResolvedValue(mockResponseStream);
      const reqParts: Part[] = [{ text: 'Hi' }];
      for await (const _ of turn.run(
        { model: 'gemini' },
        reqParts,
        new AbortController().signal,
      )) {
        // consume stream
      }
      expect(turn.getDebugResponses()).toEqual([resp1, resp2]);
    });
  });

  describe('getResponseText / collapseStreamOverlap (SOUL-SOUL_DESKTOP-264)', () => {
    // Pure-function tests for the collapse helper. The Turn-integration test
    // below covers the in-pipeline behavior via run().

    it('returns empty string for empty input', () => {
      expect(collapseStreamOverlap([])).toBe('');
    });

    it('passes a single chunk through unchanged', () => {
      expect(collapseStreamOverlap(['hello world'])).toBe('hello world');
    });

    it('joins pure deltas with a single space (back-compat)', () => {
      expect(collapseStreamOverlap(['Hello', 'world', 'again'])).toBe(
        'Hello world again',
      );
    });

    it('collapses overlap when next chunk re-emits a long suffix', () => {
      // The exact disk-pattern observed on job-hunt session 82a1b676:
      // a snapshot chunk re-emits text already streamed in chunk 0.
      const chunk0 =
        'Hello. I am ready to assist with your job search, response monitoring loop, or Soul OS kernel architecture. Let me know what we are targeting.';
      const chunk1 =
        'to assist with your job search, response monitoring loop, or Soul OS kernel architecture. Let me know what we are targeting. And here is the new tail.';
      const out = collapseStreamOverlap([chunk0, chunk1]);
      // The repeated middle should appear exactly once.
      const occurrences =
        out.split('to assist with your job search').length - 1;
      expect(occurrences).toBe(1);
      expect(out).toContain('And here is the new tail.');
      // No double-space artifact between the seam.
      expect(out).not.toContain('targeting.  to assist');
    });

    it('preserves short legitimate repeats below MIN_OVERLAP (16 chars)', () => {
      // "the the" is only 7 chars of overlap candidate; below threshold.
      // Treat as a true delta and keep both occurrences.
      const out = collapseStreamOverlap(['Click the', 'the button']);
      expect(out).toBe('Click the the button');
    });

    it('preserves repeated code-snippet tokens (} } pattern)', () => {
      // Repeated closing-brace blocks are common in generated code; the short
      // repeat between chunks must survive (overlap < MIN_OVERLAP).
      const a = 'function a() {\n  return 1;\n}';
      const b = '\n\nfunction b() {\n  return 2;\n}';
      // Overlap candidates here are at most a few chars of "}" / newline, all
      // < MIN_OVERLAP, so we expect the standard " "-separator fallback.
      const out = collapseStreamOverlap([a, b]);
      expect(out).toContain('function a()');
      expect(out).toContain('function b()');
      expect(out).toContain('return 1;');
      expect(out).toContain('return 2;');
    });

    it('handles 3-chunk overlap chain (delta, snapshot, new delta)', () => {
      const c0 = 'The quick brown fox jumps over the lazy dog every morning.';
      const c1 =
        'jumps over the lazy dog every morning. And then runs away to the forest.';
      const c2 = 'forest. Finally it rests at noon.';
      const out = collapseStreamOverlap([c0, c1, c2]);
      // Each phrase appears exactly once.
      expect(out.split('jumps over the lazy dog').length - 1).toBe(1);
      expect(out.split('to the forest').length - 1).toBe(1);
      expect(out).toContain('Finally it rests at noon.');
    });

    it('caps overlap scan at MAX_OVERLAP_SCAN (no quadratic blowup)', () => {
      // Two 50KB chunks: even if their entire bodies overlap, the scan only
      // looks at the last 4096 chars. The collapse may be partial (not
      // catastrophic) but the call must return promptly. We assert on time
      // budget rather than full collapse correctness here.
      const big = 'x'.repeat(50_000);
      const start = Date.now();
      const out = collapseStreamOverlap([big, big]);
      const elapsedMs = Date.now() - start;
      expect(elapsedMs).toBeLessThan(500);
      // With the cap, at most MAX_OVERLAP_SCAN (4096) chars are collapsed,
      // so the output length sits between |big| + (|big| - MAX_OVERLAP_SCAN)
      // and 2*|big| + 1. Just sanity-check it didn't double-output.
      expect(out.length).toBeLessThan(big.length * 2);
    });

    it('integrates with Turn: doubled stream chunks collapse in getResponseText', async () => {
      // End-to-end through Turn.run() to confirm getResponseText() sees the
      // collapsed text. Mirrors the gemini-cli stream pattern that produced
      // the SOUL-SOUL_DESKTOP-264 disk doubling.
      const chunkA =
        'Hello. I am ready to assist with your job search. Let me know what we are targeting.';
      const chunkB =
        'to assist with your job search. Let me know what we are targeting. Plus a tail.';
      const mockResponseStream = (async function* () {
        yield {
          type: StreamEventType.CHUNK,
          value: {
            candidates: [{ content: { parts: [{ text: chunkA }] } }],
          } as GenerateContentResponse,
        };
        yield {
          type: StreamEventType.CHUNK,
          value: {
            candidates: [{ content: { parts: [{ text: chunkB }] } }],
          } as GenerateContentResponse,
        };
      })();
      mockSendMessageStream.mockResolvedValue(mockResponseStream);
      const reqParts: Part[] = [{ text: 'Hi' }];
      for await (const _ of turn.run(
        { model: 'gemini' },
        reqParts,
        new AbortController().signal,
      )) {
        // consume stream
      }
      const out = turn.getResponseText();
      expect(out.split('to assist with your job search').length - 1).toBe(1);
      expect(out).toContain('Plus a tail.');
      expect(out).not.toContain('targeting.  to assist');
    });
  });
});
