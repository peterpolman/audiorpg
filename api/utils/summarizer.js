// summarizer.js
import OpenAI from "openai";
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const MODEL = process.env.SUMMARY_MODEL || "gpt-4o-mini";

export async function updateSummary({ oldSummary, recent, state }) {
  // Keep summary short & structured. Bound by ~200–300 tokens.
  const prompt = `
You compress an ongoing interactive story into a tight memory for continuity.

Rules:
- Retain only durable facts: setting, important NPCs, goals, discovered clues, promises, unresolved threads, inventory/conditions.
- Do not retell prose. Prefer bullet-like, compact sentences.
- Merge with previous summary, extend the existing bullet lists and do not remove earlier items.

=== PREVIOUS SUMMARY (may be empty) ===
${oldSummary || "(none)"}

=== NEW EXCHANGES TO FOLD IN ===
${recent
  .map((t, i) => `Turn ${i + 1} - Player: ${t.action}\nScene: ${t.scene}`)
  .join("\n\n")}

=== CURRENT STATE (hints; optional) ===
${JSON.stringify(state || {}, null, 2)}

Return only the updated summary.
  `.trim();

  const resp = await client.responses.create({
    model: MODEL,
    input: prompt,
  });
  return resp.output_text?.trim() || oldSummary || "";
}
