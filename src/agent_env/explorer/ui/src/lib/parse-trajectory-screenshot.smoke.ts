/**
 * Smoke test for the screenshot-trajectory path of `parseOtelTrajectory`:
 * OTel spans whose execute_tool results carry a base64 `screenshot`.
 *
 * Runner: plain TS, exits non-zero on assertion failure. From this package:
 *   npx tsx src/lib/parse-trajectory-screenshot.smoke.ts
 */
import {
  type OtelSpan,
  looksLikeScreenshotTrajectory,
  parseOtelTrajectory,
} from './parse-trajectory';

let failures = 0;
function assert(cond: unknown, msg: string): void {
  if (cond) {
    console.log(`✓ ${msg}`);
  } else {
    failures += 1;
    console.error(`✗ ${msg}`);
  }
}

function span(
  operation: string,
  start: string,
  attributes: Record<string, string>,
): OtelSpan {
  return {
    name: operation,
    context: { trace_id: 't', span_id: start },
    kind: 'INTERNAL',
    parent_id: null,
    start_time: start,
    end_time: start,
    status: { status_code: 'OK' },
    attributes: { 'gen_ai.operation.name': operation, ...attributes },
    events: [],
    links: [],
    resource: { attributes: {} },
  };
}

const SHOT =
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8AABQAB';

function main(): void {
  const spans = [
    span('chat', '2026-01-01T00:00:00Z', {
      'gen_ai.request.model': 'claude-test',
      'gen_ai.prompt': JSON.stringify({ prompt: 'Open the settings' }),
      'gen_ai.completion': JSON.stringify([
        { type: 'text', text: 'Clicking the gear icon' },
        { type: 'tool_use', id: 'a', name: 'click', input: { x: 1, y: 2 } },
      ]),
    }),
    span('execute_tool', '2026-01-01T00:00:01Z', {
      'gen_ai.completion': JSON.stringify({ ok: true, screenshot: SHOT }),
    }),
    span('chat', '2026-01-01T00:00:02Z', {
      'gen_ai.completion': JSON.stringify([
        { type: 'text', text: 'Settings are open.' },
      ]),
    }),
  ];

  assert(looksLikeScreenshotTrajectory(spans), 'detected by shape');
  const parsed = parseOtelTrajectory(spans);
  assert(parsed.model === 'claude-test', 'model from the first chat span');
  assert(
    parsed.userPrompt === 'Open the settings',
    'prompt from the first chat span',
  );
  assert(parsed.toolCallCount === 1, 'one tool call');
  const call = parsed.events.find(e => e.type === 'tool_call');
  assert(
    call?.type === 'tool_call' && call.result?.screenshot === SHOT,
    'screenshot attached to the tool result',
  );
  assert(
    call?.type === 'tool_call' && !call.result?.output.includes(SHOT),
    'screenshot stripped from the displayed output',
  );
  assert(
    parsed.steps.length === 1 &&
      parsed.steps[0]?.label === 'Clicking the gear icon',
    'one step per action, labelled by the preceding text',
  );
  assert(parsed.finalResponse === 'Settings are open.', 'final response');

  const plain = spans.map(s =>
    s.name === 'execute_tool'
      ? span('execute_tool', s.start_time, {
          'gen_ai.completion': JSON.stringify({ ok: true }),
        })
      : s,
  );
  assert(
    !looksLikeScreenshotTrajectory(plain),
    'no screenshot, no screenshot path',
  );
}

main();
if (failures > 0) {
  console.error(`\n${failures} assertion(s) failed.`);
  process.exit(1);
}
console.log('\nAll screenshot-trajectory smoke tests passed.');
