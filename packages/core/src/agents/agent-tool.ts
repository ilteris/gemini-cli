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
  SubagentActivityItem,
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
    if (process.env['SOUL_IS_SUBAGENT'] === '1') {
      throw new Error('Subagent recursion cap reached (depth 1).');
    }

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
      '--stream',
    ];

    const activities: SubagentActivityItem[] = [
      {
        id: activityId,
        type: 'thought',
        content: 'Routing Gemini native delegation through soul delegate.',
        status: SubagentState.RUNNING,
      },
    ];

    const getProgress = (
      state: SubagentState,
      result?: string,
    ): SubagentProgress => ({
      isSubagentProgress: true,
      agentName: this.definition.name,
      recentActivity: [...activities],
      state,
      result,
    });

    let delegationId: string | undefined;
    let findingPath: string | undefined;
    let finalSummary: string | undefined;
    let liveLog: string | undefined;

    const publishActivity = (activity: SubagentActivityItem): void => {
      void this.messageBus.publish({
        type: MessageBusType.SUBAGENT_ACTIVITY,
        subagentName: this.definition.displayName ?? this.definition.name,
        parentToolCallId: callId,
        activity: { ...activity },
      });
    };

    const updateProgress = (state: SubagentState): void => {
      updateOutput?.(getProgress(state));
    };

    interface StreamEvent {
      event: string;
      specialist?: string;
      provider?: string;
      text?: string;
      id?: string;
      name?: string;
      displayName?: string;
      args?: Record<string, unknown>;
      status?: string;
      delegation_id?: string;
      finding_path?: string;
      summary?: string;
      live_log?: string;
    }

    const handleStreamEvent = (event: StreamEvent): void => {
      if (!event || typeof event !== 'object') return;

      if (event.event === 'subagent_started') {
        delegationId = event.id || event.delegation_id;
        liveLog = event.live_log;

        const item = activities.find((a) => a.id === activityId);
        if (item) {
          item.content = `Soul delegate @${event.specialist ?? 'agent'} started on provider ${event.provider ?? 'provider'}.`;
        }
        updateProgress(SubagentState.RUNNING);
      } else if (event.event === 'subagent_completed') {
        delegationId = event.id || event.delegation_id;
        findingPath = event.finding_path;
        finalSummary = event.summary;
        updateProgress(SubagentState.COMPLETED);
      } else if (event.event === 'subagent_failed') {
        delegationId = event.id || event.delegation_id;
        updateProgress(SubagentState.ERROR);
      } else if (event.event === 'thought_chunk') {
        let chunkActivity: SubagentActivityItem;
        const last = activities[activities.length - 1];
        if (last && last.type === 'thought') {
          last.content += event.text ?? '';
          chunkActivity = last;
        } else {
          chunkActivity = {
            id: randomUUID(),
            type: 'thought',
            content: event.text ?? '',
            status: SubagentState.RUNNING,
          };
          activities.push(chunkActivity);
        }
        updateProgress(SubagentState.RUNNING);

        // Publish actual thought chunk content
        publishActivity({
          id: chunkActivity.id,
          type: 'thought',
          content: event.text ?? '',
          status: SubagentState.RUNNING,
        });
      } else if (event.event === 'tool_call_start') {
        const toolActivity: SubagentActivityItem = {
          id: event.id || randomUUID(),
          type: 'tool_call',
          content: `🔧 Running tool: ${event.displayName || event.name || 'tool'}`,
          displayName: event.displayName || event.name,
          args: event.args ? JSON.stringify(event.args) : undefined,
          status: SubagentState.RUNNING,
        };
        activities.push(toolActivity);
        updateProgress(SubagentState.RUNNING);

        // Publish the actual tool call start
        publishActivity(toolActivity);
      } else if (event.event === 'tool_call_end') {
        const item = activities.find((a) => a.id === event.id);
        if (item) {
          item.status =
            event.status === 'completed'
              ? SubagentState.COMPLETED
              : SubagentState.ERROR;
          item.content = `🔧 Tool completed: ${item.displayName || 'tool'}`;
          updateProgress(SubagentState.RUNNING);

          // Publish the updated tool call end
          publishActivity(item);
        }
      }
    };

    updateProgress(SubagentState.RUNNING);

    try {
      const output = await new Promise<string>((resolve, reject) => {
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
        let stdoutBuffer = '';

        const progressTimer = setInterval(() => {
          updateProgress(SubagentState.RUNNING);
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
          const chunkStr = chunk.toString('utf8');
          stdoutBuffer += chunkStr;

          const lines = stdoutBuffer.split('\n');
          stdoutBuffer = lines.pop() ?? ''; // Keep last unfinished line in buffer

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed) continue;

            try {
              const parsed: unknown = JSON.parse(trimmed);
              if (isRecord(parsed)) {
                const parsedEvent = parsed['event'];
                if (typeof parsedEvent === 'string') {
                  const specialistVal = parsed['specialist'];
                  const providerVal = parsed['provider'];
                  const textVal = parsed['text'];
                  const idVal = parsed['id'];
                  const nameVal = parsed['name'];
                  const displayNameVal = parsed['displayName'];
                  const argsVal = parsed['args'];
                  const statusVal = parsed['status'];
                  const delegationIdVal = parsed['delegation_id'];
                  const findingPathVal = parsed['finding_path'];
                  const summaryVal = parsed['summary'];
                  const liveLogVal = parsed['live_log'];

                  const event: StreamEvent = {
                    event: parsedEvent,
                    specialist:
                      typeof specialistVal === 'string'
                        ? specialistVal
                        : undefined,
                    provider:
                      typeof providerVal === 'string' ? providerVal : undefined,
                    text: typeof textVal === 'string' ? textVal : undefined,
                    id: typeof idVal === 'string' ? idVal : undefined,
                    name: typeof nameVal === 'string' ? nameVal : undefined,
                    displayName:
                      typeof displayNameVal === 'string'
                        ? displayNameVal
                        : undefined,
                    args: isRecord(argsVal) ? argsVal : undefined,
                    status:
                      typeof statusVal === 'string' ? statusVal : undefined,
                    delegation_id:
                      typeof delegationIdVal === 'string'
                        ? delegationIdVal
                        : undefined,
                    finding_path:
                      typeof findingPathVal === 'string'
                        ? findingPathVal
                        : undefined,
                    summary:
                      typeof summaryVal === 'string' ? summaryVal : undefined,
                    live_log:
                      typeof liveLogVal === 'string' ? liveLogVal : undefined,
                  };
                  handleStreamEvent(event);
                  continue;
                }
              }
            } catch {
              // Fallback to normal stdout accumulation if not valid JSON stream event
            }
            stdout += line + '\n';
          }
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

          const trimmed = stdoutBuffer.trim();
          if (trimmed) {
            try {
              const parsed: unknown = JSON.parse(trimmed);
              if (isRecord(parsed)) {
                const parsedEvent = parsed['event'];
                if (typeof parsedEvent === 'string') {
                  const specialistVal = parsed['specialist'];
                  const providerVal = parsed['provider'];
                  const textVal = parsed['text'];
                  const idVal = parsed['id'];
                  const nameVal = parsed['name'];
                  const displayNameVal = parsed['displayName'];
                  const argsVal = parsed['args'];
                  const statusVal = parsed['status'];
                  const delegationIdVal = parsed['delegation_id'];
                  const findingPathVal = parsed['finding_path'];
                  const summaryVal = parsed['summary'];
                  const liveLogVal = parsed['live_log'];

                  const event: StreamEvent = {
                    event: parsedEvent,
                    specialist:
                      typeof specialistVal === 'string'
                        ? specialistVal
                        : undefined,
                    provider:
                      typeof providerVal === 'string' ? providerVal : undefined,
                    text: typeof textVal === 'string' ? textVal : undefined,
                    id: typeof idVal === 'string' ? idVal : undefined,
                    name: typeof nameVal === 'string' ? nameVal : undefined,
                    displayName:
                      typeof displayNameVal === 'string'
                        ? displayNameVal
                        : undefined,
                    args: isRecord(argsVal) ? argsVal : undefined,
                    status:
                      typeof statusVal === 'string' ? statusVal : undefined,
                    delegation_id:
                      typeof delegationIdVal === 'string'
                        ? delegationIdVal
                        : undefined,
                    finding_path:
                      typeof findingPathVal === 'string'
                        ? findingPathVal
                        : undefined,
                    summary:
                      typeof summaryVal === 'string' ? summaryVal : undefined,
                    live_log:
                      typeof liveLogVal === 'string' ? liveLogVal : undefined,
                  };
                  handleStreamEvent(event);
                } else {
                  stdout += trimmed + '\n';
                }
              } else {
                stdout += trimmed + '\n';
              }
            } catch {
              stdout += trimmed + '\n';
            }
          }

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

      let summary = finalSummary || output.trim();
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

      // Turn any currently running activities to completed on finish.
      for (const act of activities) {
        if (act.status === SubagentState.RUNNING) {
          act.status = SubagentState.COMPLETED;
        }
      }

      const completed = getProgress(SubagentState.COMPLETED, summary);
      updateOutput?.(completed);

      const resultContent = `Subagent '${this.definition.name}' finished via soul delegate.
Result:
${summary}`;

      return {
        llmContent: [{ text: resultContent }],
        returnDisplay: completed,
        data: {
          agentId: callId,
          soulDelegate: true,
          specialist: this.definition.name,
          delegation_id: delegationId,
          live_log: liveLog,
          status: SubagentState.COMPLETED,
          finding_path: findingPath,
          ...(metadata ? { metadata } : {}),
        },
      };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      const isAbort = error instanceof Error && error.name === 'AbortError';
      const state = isAbort ? SubagentState.CANCELLED : SubagentState.ERROR;

      for (const act of activities) {
        if (act.status === SubagentState.RUNNING) {
          act.status = state;
        }
      }

      const failed = getProgress(state);
      updateOutput?.(failed);

      if (isAbort) {
        throw error;
      }

      return {
        llmContent: `Subagent '${this.definition.name}' failed via soul delegate. Error: ${message}`,
        returnDisplay: failed,
        data: {
          agentId: callId,
          soulDelegate: true,
          specialist: this.definition.name,
          delegation_id: delegationId,
          live_log: liveLog,
          status: state,
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
