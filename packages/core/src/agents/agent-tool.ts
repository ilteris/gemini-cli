/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  BaseDeclarativeTool,
  Kind,
  type ToolInvocation,
  type ToolResult,
  BaseToolInvocation,
  type ToolCallConfirmationDetails,
  type ExecuteOptions,
} from '../tools/tools.js';
import { type AgentLoopContext } from '../config/agent-loop-context.js';
import type { MessageBus } from '../confirmation-bus/message-bus.js';
import type {
  AgentDefinition,
  AgentInputs,
  SubagentProgress,
} from './types.js';
import { SubagentState } from './types.js';
import { LocalSubagentInvocation } from './local-invocation.js';
import { RemoteAgentInvocation } from './remote-invocation.js';
import { LocalSessionInvocation } from './local-session-invocation.js';
import { RemoteSessionInvocation } from './remote-session-invocation.js';
import { BROWSER_AGENT_NAME } from './browser/browserAgentDefinition.js';
import { BrowserAgentInvocation } from './browser/browserAgentInvocation.js';
import type { AgentEvent } from '../agent/types.js';
import { formatUserHintsForModel } from '../utils/fastAckHelper.js';
import { isRecord } from '../utils/markdownUtils.js';
import { runInDevTraceSpan } from '../telemetry/trace.js';
import {
  GeminiCliOperation,
  GEN_AI_AGENT_DESCRIPTION,
  GEN_AI_AGENT_NAME,
} from '../telemetry/constants.js';
import { AGENT_TOOL_NAME } from '../tools/tool-names.js';
import { spawn } from 'node:child_process';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { MessageBusType } from '../confirmation-bus/types.js';
import { randomUUID } from 'node:crypto';

const SOUL_DELEGATED_AGENT_NAMES = new Set([
  'adversarial_judge',
  'cloud_architect',
  'code_archaeologist',
  'creative_technologist',
  'information_retriever',
  'monorepo_architect',
  'narrative_taxonomist',
  'product_shaper',
  'registry_guardian',
  'systems_architect',
  'terrain_mapper',
  'visual_auditor',
]);

function isSoulDelegatedAgent(definition: AgentDefinition): boolean {
  if (process.env['SOUL_DELEGATE_GEMINI_NATIVE']?.toLowerCase() === 'false') {
    return false;
  }

  if (!SOUL_DELEGATED_AGENT_NAMES.has(definition.name)) {
    return false;
  }

  const soulAgentDir = path.join(
    os.homedir(),
    'dotfiles',
    'soul',
    'agents',
    definition.name,
  );
  return fs.existsSync(soulAgentDir);
}

/**
 * A unified tool for invoking subagents.
 *
 * Handles looking up the subagent, validating its eligibility,
 * mapping the general 'prompt' parameter to the agent's specific schema,
 * and delegating execution.
 */
export class AgentTool extends BaseDeclarativeTool<
  { agent_name: string; prompt: string },
  ToolResult
> {
  static readonly Name = AGENT_TOOL_NAME;

  constructor(
    private readonly context: AgentLoopContext,
    messageBus: MessageBus,
    private readonly onAgentEvent?: (event: AgentEvent) => void,
  ) {
    super(
      AGENT_TOOL_NAME,
      'Invoke Subagent',
      'Invoke a subagent to perform a specific task or investigation.',
      Kind.Agent,
      {
        type: 'object',
        properties: {
          agent_name: {
            type: 'string',
            description: 'Name of the subagent to invoke',
          },
          prompt: {
            type: 'string',
            description:
              'The COMPLETE query to send the subagent. MUST be comprehensive and detailed. Include all context, background, questions, and expected output format. Do NOT send brief or incomplete instructions.',
          },
        },
        required: ['agent_name', 'prompt'],
      },
      messageBus,
      /* isOutputMarkdown */ true,
      /* canUpdateOutput */ true,
    );
  }

  protected createInvocation(
    params: { agent_name: string; prompt: string },
    messageBus: MessageBus,
    _toolName?: string,
    _toolDisplayName?: string,
  ): ToolInvocation<{ agent_name: string; prompt: string }, ToolResult> {
    const registry = this.context.config.getAgentRegistry();
    const definition = registry.getDefinition(params.agent_name);

    if (!definition) {
      throw new Error(`Subagent '${params.agent_name}' not found.`);
    }

    // Smart Parameter Mapping
    const mappedInputs = this.mapParams(
      params.prompt,
      definition.inputConfig.inputSchema,
    );

    return new DelegateInvocation(
      params,
      mappedInputs,
      messageBus,
      definition,
      this.context,
      _toolName,
      _toolDisplayName,
      this.onAgentEvent,
    );
  }

  private mapParams(prompt: string, schema: unknown): AgentInputs {
    const schemaObj: unknown = schema;
    if (!isRecord(schemaObj)) {
      return { prompt };
    }
    const properties = schemaObj['properties'];
    if (isRecord(properties)) {
      const keys = Object.keys(properties);
      if (keys.length === 1) {
        return { [keys[0]]: prompt };
      }
    }
    return { prompt };
  }
}

class SoulDelegateInvocation extends BaseToolInvocation<
  AgentInputs,
  ToolResult
> {
  parentToolCallId?: string;

  constructor(
    private readonly definition: AgentDefinition,
    private readonly prompt: string,
    private readonly context: AgentLoopContext,
    messageBus: MessageBus,
    _toolName?: string,
    _toolDisplayName?: string,
  ) {
    super(
      { prompt },
      messageBus,
      _toolName ?? AGENT_TOOL_NAME,
      _toolDisplayName ?? `Invoke ${definition.displayName ?? definition.name}`,
    );
  }

  getDescription(): string {
    return `Delegating to agent '${this.definition.name}'`;
  }

  async execute(options: ExecuteOptions): Promise<ToolResult> {
    const { abortSignal: signal, updateOutput } = options;
    const activityId = randomUUID();
    const callId = this.parentToolCallId ?? randomUUID();
    const project = process.env['SOUL_PROJECT'] || 'global';
    const args = [
      'delegate',
      this.definition.name,
      this.prompt,
      '--project',
      project,
      '--provider',
      'gemini',
      '--mode',
      'sync',
      '--call-id',
      callId,
    ];

    const progress = (
      state: SubagentState,
      content: string,
      result?: string,
    ): SubagentProgress => ({
      isSubagentProgress: true,
      agentName: this.definition.name,
      recentActivity: [
        {
          id: activityId,
          type: 'thought',
          content,
          status: state,
        },
      ],
      state,
      result,
    });

    const publish = (state: SubagentState, content: string): void => {
      void this.messageBus.publish({
        type: MessageBusType.SUBAGENT_ACTIVITY,
        subagentName: this.definition.displayName ?? this.definition.name,
        parentToolCallId: callId,
        activity: {
          id: activityId,
          type: 'thought',
          content,
          status: state,
        },
      });
    };

    const initial = progress(
      SubagentState.RUNNING,
      'Routing Gemini native delegation through soul delegate.',
    );
    updateOutput?.(initial);
    publish(SubagentState.RUNNING, initial.recentActivity[0].content);

    try {
      const output = await new Promise<string>((resolve, reject) => {
        let elapsedSeconds = 0;
        let settled = false;
        const child = spawn('soul', args, {
          cwd: this.context.config.getProjectRoot(),
          env: {
            ...process.env,
            SOUL_SESSION_VISIBILITY: 'machine',
            SOUL_SESSION_KIND: 'subagent',
          },
          stdio: ['ignore', 'pipe', 'pipe'],
        });
        let stdout = '';
        let stderr = '';

        const progressTimer = setInterval(() => {
          elapsedSeconds += 15;
          const content = `Soul delegate still running (${elapsedSeconds}s elapsed).`;
          updateOutput?.(progress(SubagentState.RUNNING, content));
          publish(SubagentState.RUNNING, content);
        }, 15000);
        const settle = (fn: () => void): void => {
          if (settled) {
            return;
          }
          settled = true;
          clearInterval(progressTimer);
          fn();
        };

        const abort = (): void => {
          child.kill('SIGTERM');
          const error = new Error('Operation cancelled by user');
          error.name = 'AbortError';
          settle(() => reject(error));
        };

        if (signal.aborted) {
          abort();
          return;
        }

        signal.addEventListener('abort', abort, { once: true });
        child.stdout.on('data', (chunk: Buffer) => {
          stdout += chunk.toString('utf8');
        });
        child.stderr.on('data', (chunk: Buffer) => {
          stderr += chunk.toString('utf8');
        });
        child.on('error', (error) => {
          signal.removeEventListener('abort', abort);
          settle(() => reject(error));
        });
        child.on('close', (code) => {
          signal.removeEventListener('abort', abort);
          if (code === 0) {
            settle(() => resolve(stdout));
            return;
          }
          settle(() =>
            reject(
              new Error(
                stderr.trim() || `soul delegate exited with status ${code}`,
              ),
            ),
          );
        });
      });

      let summary = output.trim();
      let metadata: Record<string, unknown> | undefined;
      try {
        const parsed: unknown = JSON.parse(summary);
        if (isRecord(parsed)) {
          metadata = parsed;
          const parsedSummary = parsed['summary'];
          if (typeof parsedSummary === 'string') {
            summary = parsedSummary;
          }
        }
      } catch {
        // Non-JSON output is still a valid delegate result.
      }

      const completed = progress(
        SubagentState.COMPLETED,
        'Soul delegate completed.',
        summary,
      );
      updateOutput?.(completed);
      publish(SubagentState.COMPLETED, completed.recentActivity[0].content);

      const resultContent = `Subagent '${this.definition.name}' finished via soul delegate.
Result:
${summary}`;

      return {
        llmContent: [{ text: resultContent }],
        returnDisplay: completed,
        data: {
          agentId: callId,
          soulDelegate: true,
          ...(metadata ? { metadata } : {}),
        },
      };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      const isAbort = error instanceof Error && error.name === 'AbortError';
      const state = isAbort ? SubagentState.CANCELLED : SubagentState.ERROR;
      const failed = progress(state, message);
      updateOutput?.(failed);
      publish(state, message);

      if (isAbort) {
        throw error;
      }

      return {
        llmContent: `Subagent '${this.definition.name}' failed via soul delegate. Error: ${message}`,
        returnDisplay: failed,
        data: {
          agentId: callId,
          soulDelegate: true,
        },
      };
    }
  }
}

class DelegateInvocation extends BaseToolInvocation<
  { agent_name: string; prompt: string },
  ToolResult
> {
  private readonly startIndex: number;
  parentToolCallId?: string;

  constructor(
    params: { agent_name: string; prompt: string },
    private readonly mappedInputs: AgentInputs,
    messageBus: MessageBus,
    private readonly definition: AgentDefinition,
    private readonly context: AgentLoopContext,
    _toolName?: string,
    _toolDisplayName?: string,
    private readonly onAgentEvent?: (event: AgentEvent) => void,
  ) {
    super(
      params,
      messageBus,
      _toolName ?? AGENT_TOOL_NAME,
      _toolDisplayName ?? `Invoke ${definition.displayName ?? definition.name}`,
    );
    this.startIndex = context.config.injectionService.getLatestInjectionIndex();
  }

  getDescription(): string {
    return `Delegating to agent '${this.definition.name}'`;
  }

  private buildChildInvocation(
    agentArgs: AgentInputs,
  ): ToolInvocation<AgentInputs, ToolResult> {
    if (this.definition.name === BROWSER_AGENT_NAME) {
      return new BrowserAgentInvocation(
        this.context,
        agentArgs,
        this.messageBus,
        this._toolName,
        this._toolDisplayName,
      );
    }

    if (isSoulDelegatedAgent(this.definition)) {
      return new SoulDelegateInvocation(
        this.definition,
        this.params.prompt,
        this.context,
        this.messageBus,
        this._toolName,
        this._toolDisplayName,
      );
    }

    const useSession = this.context.config.isAgentSessionSubagentEnabled();
    const options = this.onAgentEvent
      ? { onAgentEvent: this.onAgentEvent }
      : undefined;

    if (this.definition.kind === 'remote') {
      if (useSession) {
        return new RemoteSessionInvocation(
          this.definition,
          this.context,
          agentArgs,
          this.messageBus,
          options,
        );
      }
      return new RemoteAgentInvocation(
        this.definition,
        this.context,
        agentArgs,
        this.messageBus,
      );
    } else {
      if (useSession) {
        return new LocalSessionInvocation(
          this.definition,
          this.context,
          agentArgs,
          this.messageBus,
          options,
        );
      }
      return new LocalSubagentInvocation(
        this.definition,
        this.context,
        agentArgs,
        this.messageBus,
      );
    }
  }

  override async shouldConfirmExecute(
    abortSignal: AbortSignal,
  ): Promise<ToolCallConfirmationDetails | false> {
    const hintedParams = this.withUserHints(this.mappedInputs);
    const invocation = this.buildChildInvocation(hintedParams);
    Object.assign(invocation, { parentToolCallId: this.parentToolCallId });
    return invocation.shouldConfirmExecute(abortSignal);
  }

  async execute(options: ExecuteOptions): Promise<ToolResult> {
    const { abortSignal: signal, updateOutput } = options;
    const hintedParams = this.withUserHints(this.mappedInputs);
    const invocation = this.buildChildInvocation(hintedParams);
    Object.assign(invocation, { parentToolCallId: this.parentToolCallId });

    return runInDevTraceSpan(
      {
        operation: GeminiCliOperation.AgentCall,
        logPrompts: this.context.config.getTelemetryLogPromptsEnabled(),
        tracesEnabled: this.context.config.getTelemetryTracesEnabled(),
        sessionId: this.context.config.getSessionId(),
        attributes: {
          [GEN_AI_AGENT_NAME]: this.definition.name,
          [GEN_AI_AGENT_DESCRIPTION]: this.definition.description,
        },
      },
      async ({ metadata }) => {
        metadata.input = this.params;
        const result = await invocation.execute({
          abortSignal: signal,
          updateOutput,
        });
        metadata.output = result;
        return result;
      },
    );
  }

  private withUserHints(agentArgs: AgentInputs): AgentInputs {
    if (this.definition.kind !== 'remote') {
      return agentArgs;
    }

    const userHints = this.context.config.injectionService.getInjectionsAfter(
      this.startIndex,
      'user_steering',
    );
    const formattedHints = formatUserHintsForModel(userHints);
    if (!formattedHints) {
      return agentArgs;
    }

    // Find the primary key to append hints to
    const schemaObj: unknown = this.definition.inputConfig.inputSchema;
    if (!isRecord(schemaObj)) {
      return agentArgs;
    }
    const properties = schemaObj['properties'];
    if (isRecord(properties)) {
      const keys = Object.keys(properties);
      const primaryKey = keys.length === 1 ? keys[0] : 'prompt';

      const value = agentArgs[primaryKey];
      if (typeof value !== 'string' || value.trim().length === 0) {
        return agentArgs;
      }

      return {
        ...agentArgs,
        [primaryKey]: `${formattedHints}\n\n${value}`,
      };
    }

    return agentArgs;
  }
}
