/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { AgentTool } from './agent-tool.js';
import { makeFakeConfig } from '../test-utils/config.js';
import { createMockMessageBus } from '../test-utils/mock-message-bus.js';
import type { Config } from '../config/config.js';
import type { MessageBus } from '../confirmation-bus/message-bus.js';
import { LocalSubagentInvocation } from './local-invocation.js';
import { RemoteAgentInvocation } from './remote-invocation.js';
import { LocalSessionInvocation } from './local-session-invocation.js';
import { RemoteSessionInvocation } from './remote-session-invocation.js';
import { BrowserAgentInvocation } from './browser/browserAgentInvocation.js';
import { BROWSER_AGENT_NAME } from './browser/browserAgentDefinition.js';
import { AgentRegistry } from './registry.js';
import type { LocalAgentDefinition, RemoteAgentDefinition } from './types.js';

vi.mock('./local-invocation.js');
vi.mock('./remote-invocation.js');
vi.mock('./local-session-invocation.js');
vi.mock('./remote-session-invocation.js');
vi.mock('./browser/browserAgentInvocation.js');

vi.mock('node:fs', async (importOriginal) => {
  const actual = await importOriginal<typeof import('node:fs')>();
  return {
    ...actual,
    existsSync: vi.fn().mockImplementation((path) => {
      if (path.toString().includes('adversarial_judge')) return true;
      return actual.existsSync(path);
    }),
  };
});

vi.mock('node:child_process', async (importOriginal) => {
  const actual = await importOriginal<typeof import('node:child_process')>();
  return {
    ...actual,
    spawn: vi.fn(),
  };
});

describe('AgentTool', () => {
  let mockConfig: Config;
  let mockMessageBus: MessageBus;
  let tool: AgentTool;

  const testLocalDefinition: LocalAgentDefinition = {
    kind: 'local',
    name: 'TestLocalAgent',
    description: 'A local test agent.',
    inputConfig: {
      inputSchema: {
        type: 'object',
        properties: { objective: { type: 'string' } },
      },
    },
    modelConfig: { model: 'test', generateContentConfig: {} },
    runConfig: { maxTimeMinutes: 1 },
    promptConfig: { systemPrompt: 'test' },
  };

  const testRemoteDefinition: RemoteAgentDefinition = {
    kind: 'remote',
    name: 'TestRemoteAgent',
    description: 'A remote test agent.',
    inputConfig: {
      inputSchema: {
        type: 'object',
        properties: { query: { type: 'string' } },
      },
    },
    agentCardUrl: 'http://example.com/agent',
  };

  beforeEach(() => {
    vi.clearAllMocks();
    mockConfig = makeFakeConfig();
    mockMessageBus = createMockMessageBus();
    tool = new AgentTool(mockConfig, mockMessageBus);

    // Mock AgentRegistry
    const registry = new AgentRegistry(mockConfig);
    vi.spyOn(mockConfig, 'getAgentRegistry').mockReturnValue(registry);

    vi.spyOn(registry, 'getDefinition').mockImplementation((name: string) => {
      if (name === 'TestLocalAgent') return testLocalDefinition;
      if (name === 'TestRemoteAgent') return testRemoteDefinition;
      if (name === BROWSER_AGENT_NAME) {
        return {
          kind: 'remote',
          name: BROWSER_AGENT_NAME,
          displayName: 'Browser Agent',
          description: 'Browser Agent Description',
          inputConfig: {
            inputSchema: {
              type: 'object',
              properties: { task: { type: 'string' } },
            },
          },
          agentCardUrl: 'http://example.com',
        };
      }
      return undefined;
    });
  });

  it('should map prompt to objective for local agent', async () => {
    const params = { agent_name: 'TestLocalAgent', prompt: 'Do something' };
    const invocation = tool['createInvocation'](params, mockMessageBus);

    // Trigger deferred instantiation
    await invocation.shouldConfirmExecute(new AbortController().signal);

    expect(LocalSubagentInvocation).toHaveBeenCalledWith(
      testLocalDefinition,
      mockConfig,
      { objective: 'Do something' },
      mockMessageBus,
    );
  });

  it('should map prompt to query for remote agent', async () => {
    const params = {
      agent_name: 'TestRemoteAgent',
      prompt: 'Search something',
    };
    const invocation = tool['createInvocation'](params, mockMessageBus);

    // Trigger deferred instantiation
    await invocation.shouldConfirmExecute(new AbortController().signal);

    expect(RemoteAgentInvocation).toHaveBeenCalledWith(
      testRemoteDefinition,
      mockConfig,
      { query: 'Search something' },
      mockMessageBus,
    );
  });

  it('should throw error for unknown subagent', () => {
    const params = { agent_name: 'UnknownAgent', prompt: 'Hello' };
    expect(() => {
      tool['createInvocation'](params, mockMessageBus);
    }).toThrow("Subagent 'UnknownAgent' not found.");
  });

  it('should map prompt to task and use BrowserAgentInvocation for browser agent', async () => {
    const params = { agent_name: BROWSER_AGENT_NAME, prompt: 'Open page' };
    const invocation = tool['createInvocation'](params, mockMessageBus);

    // Trigger deferred instantiation
    await invocation.shouldConfirmExecute(new AbortController().signal);

    expect(BrowserAgentInvocation).toHaveBeenCalledWith(
      mockConfig,
      { task: 'Open page' },
      mockMessageBus,
      'invoke_agent',
      'Invoke Browser Agent',
    );
  });

  describe('agentSessionSubagentEnabled feature flag', () => {
    it('should use LocalSessionInvocation when flag is enabled for local agent', async () => {
      vi.spyOn(mockConfig, 'isAgentSessionSubagentEnabled').mockReturnValue(
        true,
      );
      tool = new AgentTool(mockConfig, mockMessageBus);

      const params = {
        agent_name: 'TestLocalAgent',
        prompt: 'Do something',
      };
      const invocation = tool['createInvocation'](params, mockMessageBus);
      await invocation.shouldConfirmExecute(new AbortController().signal);

      expect(LocalSessionInvocation).toHaveBeenCalledWith(
        testLocalDefinition,
        mockConfig,
        { objective: 'Do something' },
        mockMessageBus,
        undefined,
      );
      expect(LocalSubagentInvocation).not.toHaveBeenCalled();
    });

    it('should use RemoteSessionInvocation when flag is enabled for remote agent', async () => {
      vi.spyOn(mockConfig, 'isAgentSessionSubagentEnabled').mockReturnValue(
        true,
      );
      tool = new AgentTool(mockConfig, mockMessageBus);

      const params = {
        agent_name: 'TestRemoteAgent',
        prompt: 'Search something',
      };
      const invocation = tool['createInvocation'](params, mockMessageBus);
      await invocation.shouldConfirmExecute(new AbortController().signal);

      expect(RemoteSessionInvocation).toHaveBeenCalledWith(
        testRemoteDefinition,
        mockConfig,
        { query: 'Search something' },
        mockMessageBus,
        undefined,
      );
      expect(RemoteAgentInvocation).not.toHaveBeenCalled();
    });

    it('should use legacy invocations when flag is disabled (default)', async () => {
      vi.spyOn(mockConfig, 'isAgentSessionSubagentEnabled').mockReturnValue(
        false,
      );
      tool = new AgentTool(mockConfig, mockMessageBus);

      const localParams = {
        agent_name: 'TestLocalAgent',
        prompt: 'Do something',
      };
      const localInv = tool['createInvocation'](localParams, mockMessageBus);
      await localInv.shouldConfirmExecute(new AbortController().signal);

      expect(LocalSubagentInvocation).toHaveBeenCalled();
      expect(LocalSessionInvocation).not.toHaveBeenCalled();

      vi.clearAllMocks();

      const remoteParams = {
        agent_name: 'TestRemoteAgent',
        prompt: 'Search',
      };
      const remoteInv = tool['createInvocation'](remoteParams, mockMessageBus);
      await remoteInv.shouldConfirmExecute(new AbortController().signal);

      expect(RemoteAgentInvocation).toHaveBeenCalled();
      expect(RemoteSessionInvocation).not.toHaveBeenCalled();
    });

    it('should thread onAgentEvent to session invocations', async () => {
      vi.spyOn(mockConfig, 'isAgentSessionSubagentEnabled').mockReturnValue(
        true,
      );
      const onEvent = vi.fn();
      tool = new AgentTool(mockConfig, mockMessageBus, onEvent);

      const params = {
        agent_name: 'TestLocalAgent',
        prompt: 'Do something',
      };
      const invocation = tool['createInvocation'](params, mockMessageBus);
      await invocation.shouldConfirmExecute(new AbortController().signal);

      expect(LocalSessionInvocation).toHaveBeenCalledWith(
        testLocalDefinition,
        mockConfig,
        { objective: 'Do something' },
        mockMessageBus,
        { onAgentEvent: onEvent },
      );
    });

    it('should always use BrowserAgentInvocation for browser agent regardless of flag', async () => {
      vi.spyOn(mockConfig, 'isAgentSessionSubagentEnabled').mockReturnValue(
        true,
      );
      tool = new AgentTool(mockConfig, mockMessageBus);

      const params = {
        agent_name: BROWSER_AGENT_NAME,
        prompt: 'Open page',
      };
      const invocation = tool['createInvocation'](params, mockMessageBus);
      await invocation.shouldConfirmExecute(new AbortController().signal);

      expect(BrowserAgentInvocation).toHaveBeenCalled();
      expect(LocalSessionInvocation).not.toHaveBeenCalled();
      expect(RemoteSessionInvocation).not.toHaveBeenCalled();
    });
  });

  describe('Subagent Recursion and Environment Constraints', () => {
    it('should throw an error during createInvocation if SOUL_IS_SUBAGENT === 1 (recursion cap)', () => {
      vi.stubEnv('SOUL_IS_SUBAGENT', '1');
      tool = new AgentTool(mockConfig, mockMessageBus);

      const params = {
        agent_name: 'TestLocalAgent',
        prompt: 'Do something recursive',
      };
      expect(() => {
        tool['createInvocation'](params, mockMessageBus);
      }).toThrow('Subagent recursion cap reached (depth 1).');

      vi.unstubAllEnvs();
    });
  });

  /* eslint-disable @typescript-eslint/no-explicit-any */
  describe('SoulDelegateInvocation', () => {
    it('should stream child process stdout, publish structured activity, and construct metadata', async () => {
      // Mock child process spawn
      const { spawn } = await import('node:child_process');
      const { EventEmitter } = await import('node:events');

      const mockChild = new EventEmitter() as any;
      mockChild.stdout = new EventEmitter();
      mockChild.stderr = new EventEmitter();
      mockChild.kill = vi.fn();

      vi.mocked(spawn).mockReturnValue(mockChild);

      // Create a soul delegated agent definition
      const registry = mockConfig.getAgentRegistry();
      const soulDefinition = {
        kind: 'local' as const,
        name: 'adversarial_judge',
        description: 'Test Soul Agent',
        inputConfig: {
          inputSchema: {
            type: 'object',
            properties: { prompt: { type: 'string' } },
          },
        },
        modelConfig: { model: 'test', generateContentConfig: {} },
        runConfig: { maxTimeMinutes: 1 },
        promptConfig: { systemPrompt: 'test' },
      };

      vi.spyOn(registry, 'getDefinition').mockReturnValue(soulDefinition);

      const params = {
        agent_name: 'adversarial_judge',
        prompt: 'Audit this code',
      };
      const invocation = tool['createInvocation'](params, mockMessageBus);

      // Execute in background/sync mode
      const execPromise = invocation.execute({
        abortSignal: new AbortController().signal,
        updateOutput: vi.fn(),
      });

      // Emit simulated streaming JSON events
      const eventsList = [
        {
          event: 'subagent_started',
          delegation_id: 'del-123',
          specialist: 'adversarial_judge',
          provider: 'gemini',
          live_log: 'live.log',
        },
        { event: 'thought_chunk', text: 'Analyzing ' },
        { event: 'thought_chunk', text: 'correctness.' },
        {
          event: 'tool_call_start',
          id: 'tool-abc',
          name: 'read_file',
          displayName: 'Read File',
          args: { path: 'file.txt' },
        },
        { event: 'tool_call_end', id: 'tool-abc', status: 'completed' },
        {
          event: 'subagent_completed',
          delegation_id: 'del-123',
          summary: 'Audit completed successfully.',
        },
      ];

      for (const ev of eventsList) {
        mockChild.stdout.emit('data', Buffer.from(JSON.stringify(ev) + '\n'));
      }

      // Close the process
      mockChild.emit('close', 0);

      const result = await execPromise;

      // Verify final outputs and metadata
      expect(result.llmContent).toEqual([
        {
          text: "Subagent 'adversarial_judge' finished via soul delegate.\nResult:\nAudit completed successfully.",
        },
      ]);
      expect(result.data).toMatchObject({
        soulDelegate: true,
        specialist: 'adversarial_judge',
        delegation_id: 'del-123',
        live_log: 'live.log',
        finding_path: undefined,
        status: 'completed',
      });

      // Verify that SUBAGENT_ACTIVITY messages were published
      // It should NOT publish any 'thought' messages containing "Running tool" or "Completed tool" or "Soul delegate still running"
      const published = vi
        .mocked(mockMessageBus.publish)
        .mock.calls.map((c) => c[0]);

      // Filter published activity content
      const publishedThoughts = published
        .filter(
          (p) =>
            p.type === 'subagent-activity' &&
            (p as any).activity.type === 'thought',
        )
        .map((p) => (p as any).activity.content);

      expect(publishedThoughts).toContain('Analyzing ');
      expect(publishedThoughts).toContain('correctness.');

      // Ensure no prose leakage / status lines were published as thoughts
      for (const t of publishedThoughts) {
        expect(t).not.toContain('Running tool:');
        expect(t).not.toContain('Completed tool:');
        expect(t).not.toContain('Soul delegate still running');
      }

      // Ensure tool call start/end were published as 'tool_call' type, not 'thought'
      const publishedTools = published
        .filter(
          (p) =>
            p.type === 'subagent-activity' &&
            (p as any).activity.type === 'tool_call',
        )
        .map((p) => (p as any).activity);

      expect(publishedTools).toHaveLength(2); // 1 start, 1 end
      expect(publishedTools[0]).toMatchObject({
        id: 'tool-abc',
        type: 'tool_call',
        content: '🔧 Running tool: Read File',
        status: 'running',
      });
      expect(publishedTools[1]).toMatchObject({
        id: 'tool-abc',
        type: 'tool_call',
        content: '🔧 Tool completed: Read File',
        status: 'completed',
      });
    });
  });
  /* eslint-enable @typescript-eslint/no-explicit-any */
});
