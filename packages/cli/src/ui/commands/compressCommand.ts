/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { MessageType, type HistoryItemCompression } from '../types.js';
import { CommandKind, type SlashCommand } from './types.js';
import { exec } from 'node:child_process';
import { promisify } from 'node:util';

const execAsync = promisify(exec);

export const compressCommand: SlashCommand = {
  name: 'compress',
  altNames: ['summarize', 'compact'],
  description: 'Compresses the context by replacing it with a summary',
  kind: CommandKind.BUILT_IN,
  autoExecute: true,
  action: async (context) => {
    const { ui } = context;

    const isManualCompact = context.invocation?.name === 'compact';
    if (isManualCompact) {
      try {
        const { stdout } = await execAsync(
          'soul compact --provider geminiCLI --json',
        );
        // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
        const policyResult = JSON.parse(stdout.trim()) as {
          action?: string;
          usage_pct?: number;
          reason?: string;
        };
        if (policyResult.action === 'skip') {
          const pct = Math.round((policyResult.usage_pct || 0) * 100);
          const reason = policyResult.reason || 'below_threshold';
          ui.addItem(
            {
              type: MessageType.INFO,
              text: `Context is at ${pct}%. Compaction skipped (${reason === 'below_threshold' ? 'below threshold of 50%' : reason}).`,
            },
            Date.now(),
          );
          return;
        }
      } catch {
        // Fallback silently if soul compact check fails (e.g. soul command not found)
      }
    }
    if (ui.pendingItem) {
      ui.addItem(
        {
          type: MessageType.ERROR,
          text: 'Already compressing, wait for previous request to complete',
        },
        Date.now(),
      );
      return;
    }

    const pendingMessage: HistoryItemCompression = {
      type: MessageType.COMPRESSION,
      compression: {
        isPending: true,
        originalTokenCount: null,
        newTokenCount: null,
        compressionStatus: null,
      },
    };

    ui.setPendingItem(pendingMessage);

    void (async () => {
      try {
        const promptId = `compress-${Date.now()}`;
        const compressed =
          await context.services.agentContext?.geminiClient?.tryCompressChat(
            promptId,
            true,
          );
        if (compressed) {
          ui.addItem(
            {
              type: MessageType.COMPRESSION,
              compression: {
                isPending: false,
                originalTokenCount: compressed.originalTokenCount,
                newTokenCount: compressed.newTokenCount,
                compressionStatus: compressed.compressionStatus,
              },
            } as HistoryItemCompression,
            Date.now(),
          );
        } else {
          ui.addItem(
            {
              type: MessageType.ERROR,
              text: 'Failed to compress chat history.',
            },
            Date.now(),
          );
        }
      } catch (e) {
        ui.addItem(
          {
            type: MessageType.ERROR,
            text: `Failed to compress chat history: ${
              e instanceof Error ? e.message : String(e)
            }`,
          },
          Date.now(),
        );
      } finally {
        ui.setPendingItem(null);
      }
    })();
  },
};
