/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { randomUUID } from 'node:crypto';

export function createSessionId(): string {
  const env = process.env['GEMINI_SESSION_ID'];
  if (
    env &&
    /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(env)
  ) {
    return env.toLowerCase();
  }
  return randomUUID();
}
