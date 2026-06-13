/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  CompressionStatus,
  type ChatCompressionInfo,
  type GeminiClient,
} from '@google/gemini-cli-core';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import { compressCommand } from './compressCommand.js';
import { exec } from 'node:child_process';
import { promisify } from 'node:util';
import { createMockCommandContext } from '../../test-utils/mockCommandContext.js';
import { MessageType } from '../types.js';

// Mock child_process entirely inside the factory (no out-of-scope variables referenced)
vi.mock('node:child_process', () => {
  const mockExecPromise = vi.fn();
  const execMock = vi.fn();
  (execMock as unknown as Record<string | symbol, unknown>)[
    Symbol.for('nodejs.util.promisify.custom')
  ] = mockExecPromise;
  return { exec: execMock };
});

const execAsync = promisify(exec);

describe('compressCommand', () => {
  let context: ReturnType<typeof createMockCommandContext>;
  let mockTryCompressChat: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    mockTryCompressChat = vi.fn();
    context = createMockCommandContext({
      services: {
        agentContext: {
          geminiClient: {
            tryCompressChat: mockTryCompressChat,
          } as unknown as GeminiClient,
        },
      },
    });
    vi.mocked(execAsync).mockReset();
  });

  it('should do nothing if a compression is already pending', async () => {
    context.ui.pendingItem = {
      type: MessageType.COMPRESSION,
      compression: {
        isPending: true,
        originalTokenCount: null,
        newTokenCount: null,
        compressionStatus: null,
      },
    };
    await compressCommand.action!(context, '');
    await new Promise((r) => setTimeout(r, 0));
    expect(context.ui.addItem).toHaveBeenCalledWith(
      expect.objectContaining({
        type: MessageType.ERROR,
        text: 'Already compressing, wait for previous request to complete',
      }),
      expect.any(Number),
    );
    expect(context.ui.setPendingItem).not.toHaveBeenCalled();
    expect(mockTryCompressChat).not.toHaveBeenCalled();
  });

  it('should set pending item, call tryCompressChat, and add result on success', async () => {
    const compressedResult: ChatCompressionInfo = {
      originalTokenCount: 200,
      compressionStatus: CompressionStatus.COMPRESSED,
      newTokenCount: 100,
    };
    mockTryCompressChat.mockResolvedValue(compressedResult);

    await compressCommand.action!(context, '');
    await new Promise((r) => setTimeout(r, 0));

    expect(context.ui.setPendingItem).toHaveBeenNthCalledWith(1, {
      type: MessageType.COMPRESSION,
      compression: {
        isPending: true,
        compressionStatus: null,
        originalTokenCount: null,
        newTokenCount: null,
      },
    });

    expect(mockTryCompressChat).toHaveBeenCalledWith(
      expect.stringMatching(/^compress-\d+$/),
      true,
    );

    expect(context.ui.addItem).toHaveBeenCalledWith(
      {
        type: MessageType.COMPRESSION,
        compression: {
          isPending: false,
          compressionStatus: CompressionStatus.COMPRESSED,
          originalTokenCount: 200,
          newTokenCount: 100,
        },
      },
      expect.any(Number),
    );

    expect(context.ui.setPendingItem).toHaveBeenNthCalledWith(2, null);
  });

  it('should add an error message if tryCompressChat returns falsy', async () => {
    mockTryCompressChat.mockResolvedValue(null);

    await compressCommand.action!(context, '');
    await new Promise((r) => setTimeout(r, 0));

    expect(context.ui.addItem).toHaveBeenCalledWith(
      expect.objectContaining({
        type: MessageType.ERROR,
        text: 'Failed to compress chat history.',
      }),
      expect.any(Number),
    );
    expect(context.ui.setPendingItem).toHaveBeenCalledWith(null);
  });

  it('should add an error message if tryCompressChat throws', async () => {
    const error = new Error('Compression failed');
    mockTryCompressChat.mockRejectedValue(error);

    await compressCommand.action!(context, '');
    await new Promise((r) => setTimeout(r, 0));

    expect(context.ui.addItem).toHaveBeenCalledWith(
      expect.objectContaining({
        type: MessageType.ERROR,
        text: `Failed to compress chat history: ${error.message}`,
      }),
      expect.any(Number),
    );
    expect(context.ui.setPendingItem).toHaveBeenCalledWith(null);
  });

  it('should clear the pending item in a finally block', async () => {
    mockTryCompressChat.mockRejectedValue(new Error('some error'));
    await compressCommand.action!(context, '');
    await new Promise((r) => setTimeout(r, 0));
    expect(context.ui.setPendingItem).toHaveBeenCalledWith(null);
  });

  describe('metadata', () => {
    it('should have the correct name and aliases', () => {
      expect(compressCommand.name).toBe('compress');
      expect(compressCommand.altNames).toContain('summarize');
      expect(compressCommand.altNames).toContain('compact');
    });
  });

  describe('compact alias checking logic', () => {
    it('should skip compaction if soul compact skips', async () => {
      vi.mocked(execAsync).mockResolvedValue({
        stdout: JSON.stringify({
          action: 'skip',
          usage_pct: 0.22,
          reason: 'below_threshold',
        }),
        stderr: '',
      });

      context.invocation = {
        raw: '/compact',
        name: 'compact',
        args: '',
      };

      await compressCommand.action!(context, '');
      await new Promise((r) => setTimeout(r, 0));

      expect(context.ui.addItem).toHaveBeenCalledWith(
        expect.objectContaining({
          type: MessageType.INFO,
          text: 'Context is at 22%. Compaction skipped (below threshold of 50%).',
        }),
        expect.any(Number),
      );
      expect(mockTryCompressChat).not.toHaveBeenCalled();
    });

    it('should run compaction if soul compact recommends it', async () => {
      vi.mocked(execAsync).mockResolvedValue({
        stdout: JSON.stringify({ action: 'send_slash', command: '/compress' }),
        stderr: '',
      });

      mockTryCompressChat.mockResolvedValue({
        originalTokenCount: 200,
        compressionStatus: CompressionStatus.COMPRESSED,
        newTokenCount: 100,
      });

      context.invocation = {
        raw: '/compact',
        name: 'compact',
        args: '',
      };

      await compressCommand.action!(context, '');
      await new Promise((r) => setTimeout(r, 0));

      expect(mockTryCompressChat).toHaveBeenCalled();
    });
  });
});
