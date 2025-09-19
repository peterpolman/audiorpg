import "dotenv/config";
import express from "express";
import OpenAI from "openai";
import { getOrCreateSession } from "./sessions.js";
import { updateSummary } from "./summarizer.js";
// --- Direct Whisper STT using Transformers.js ---
import { pipeline } from "@xenova/transformers";
import { spawn } from "child_process";
import multer from "multer";
import { fileURLToPath } from "url";
import wavDecoder from "wav-decoder";

const __filename = fileURLToPath(import.meta.url);

const app = express();
app.use(express.json());
app.use(express.static("public"));

const upload = multer(); // memory storage
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const MODEL = process.env.MODEL || "gpt-4o-mini";

// Initialize Whisper model (lazy loading)
let whisperPipeline = null;
const WHISPER_MODEL = process.env.WHISPER_MODEL || "Xenova/whisper-base.en";

async function getWhisperPipeline() {
  if (!whisperPipeline) {
    console.log(`Loading Whisper model: ${WHISPER_MODEL}...`);
    whisperPipeline = await pipeline(
      "automatic-speech-recognition",
      WHISPER_MODEL
    );
    console.log("Whisper model loaded successfully!");
  }
  return whisperPipeline;
}

// Convert audio to WAV format using ffmpeg
function convertToWav(inputBuffer) {
  return new Promise((resolve, reject) => {
    const ffmpeg = spawn("ffmpeg", [
      "-i",
      "pipe:0", // Read from stdin
      "-f",
      "wav", // Output format
      "-acodec",
      "pcm_s16le", // Audio codec
      "-ar",
      "16000", // Sample rate (16kHz for Whisper)
      "-ac",
      "1", // Mono channel
      "-af",
      "highpass=f=80,lowpass=f=8000,volume=2.0", // Audio filters: remove noise, boost volume
      "-t",
      "30", // Limit to 30 seconds max
      "pipe:1", // Output to stdout
    ]);

    let outputBuffer = Buffer.alloc(0);
    let errorOutput = "";

    ffmpeg.stdout.on("data", (chunk) => {
      outputBuffer = Buffer.concat([outputBuffer, chunk]);
    });

    ffmpeg.stderr.on("data", (chunk) => {
      errorOutput += chunk.toString();
    });

    ffmpeg.on("close", (code) => {
      if (code === 0) {
        resolve(outputBuffer);
      } else {
        reject(new Error(`FFmpeg failed with code ${code}: ${errorOutput}`));
      }
    });

    ffmpeg.on("error", (err) => {
      reject(new Error(`FFmpeg error: ${err.message}`));
    });

    // Write input data and close stdin
    ffmpeg.stdin.write(inputBuffer);
    ffmpeg.stdin.end();
  });
}

app.post("/stt", upload.single("audio"), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: "No audio file" });

    console.log("Processing audio file...");

    // Convert WebM to WAV using ffmpeg
    const wavBuffer = await convertToWav(req.file.buffer);

    // Decode WAV to get audio data
    const audioData = await wavDecoder.decode(wavBuffer);

    // Convert to Float32Array (required format for Transformers.js)
    const audioArray = new Float32Array(audioData.channelData[0]);

    console.log("Audio converted, running transcription...");

    // Get Whisper pipeline and transcribe
    const transcriber = await getWhisperPipeline();
    const result = await transcriber(audioArray, {
      sampling_rate: audioData.sampleRate,
    });

    console.log("Transcription result:", result.text);
    res.json({ text: result.text });
  } catch (e) {
    console.error("STT Error:", e);
    res.status(500).json({ error: e.message || "STT processing failed" });
  }
});

// Combined endpoint: STT + Story Generation + TTS (LOCAL STT + OpenAI TTS)
app.post("/audio-to-story", upload.single("audio"), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: "No audio file" });

    // Get additional data from form fields
    const sessionId = req.body.sessionId;
    const character = req.body.character
      ? JSON.parse(req.body.character)
      : null;

    if (!sessionId || !character) {
      return res.status(400).json({ error: "Missing sessionId or character" });
    }

    // SSE headers
    res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
    res.setHeader("Cache-Control", "no-cache, no-transform");
    res.setHeader("Connection", "keep-alive");

    sseWrite(res, { type: "status", message: "Processing audio..." });

    // Step 1: Convert and transcribe audio (LOCAL ONLY)
    const wavBuffer = await convertToWav(req.file.buffer);
    const audioData = await wavDecoder.decode(wavBuffer);
    const audioArray = new Float32Array(audioData.channelData[0]);

    sseWrite(res, { type: "status", message: "Transcribing locally..." });

    const transcriber = await getWhisperPipeline();

    const transcriptionResult = await transcriber(audioArray, {
      sampling_rate: audioData.sampleRate,
      language: "english", // Force English for better accuracy
      task: "transcribe", // Explicitly set task
      return_timestamps: false, // We don't need timestamps
      chunk_length_s: 30, // Process in 30-second chunks
      stride_length_s: 5, // Overlap between chunks
    });

    const action = transcriptionResult.text.trim();

    console.log("Local transcription result:", action);
    if (!action || action.length < 2) {
      sseWrite(res, {
        type: "error",
        message: "Could not understand audio clearly",
      });
      res.end();
      return;
    }

    sseWrite(res, { type: "transcription", text: action });
    sseWrite(res, { type: "status", message: "Generating story..." });

    // Step 2: Generate story with streaming
    const session = getOrCreateSession(sessionId);
    const prompt = buildPrompt({
      character,
      action,
      summary: session.summary,
      lastScene: session.lastScene,
      recent: session.recent,
    });

    let fullScene = "";
    let ttsBuffer = ""; // Buffer for TTS processing

    const stream = await client.responses.create({
      model: MODEL,
      input: prompt,
      stream: true,
    });

    for await (const event of stream) {
      if (event.type === "response.output_text.delta") {
        fullScene += event.delta;
        ttsBuffer += event.delta;

        // Send text delta for immediate display
        sseWrite(res, { type: "delta", text: event.delta });

        // Check if we have complete sentences for TTS
        const sentences = extractCompleteSentences(ttsBuffer);
        if (sentences.complete) {
          // Generate TTS audio for complete sentences
          try {
            console.log("Generating TTS for:", sentences.complete);
            const ttsResponse = await client.audio.speech.create({
              model: "gpt-4o-mini-tts",
              voice: "alloy",
              input: sentences.complete,
              response_format: "mp3",
            });

            const audioBuffer = await ttsResponse.arrayBuffer();
            const audioBase64 = Buffer.from(audioBuffer).toString("base64");

            // Send audio chunk to client
            sseWrite(res, {
              type: "audio",
              audio: audioBase64,
              format: "mp3",
            });
          } catch (ttsError) {
            console.error("TTS Error:", ttsError);
            // Continue without TTS if it fails
          }

          ttsBuffer = sentences.remaining;
        }
      } else if (event.type === "response.error") {
        sseWrite(res, {
          type: "error",
          message: event.error?.message || "OpenAI error",
        });
      } else if (event.type === "response.completed") {
        // Process any remaining text for TTS
        if (ttsBuffer.trim()) {
          try {
            const ttsResponse = await client.audio.speech.create({
              model: "tts-1",
              voice: "alloy",
              input: ttsBuffer.trim(),
              response_format: "mp3",
            });

            const audioBuffer = await ttsResponse.arrayBuffer();
            const audioBase64 = Buffer.from(audioBuffer).toString("base64");

            sseWrite(res, {
              type: "audio",
              audio: audioBase64,
              format: "mp3",
            });
          } catch (ttsError) {
            console.error("TTS Error for remaining text:", ttsError);
          }
        }

        sseWrite(res, { type: "done" });
      }
    }

    // Update session memory
    session.recent.push({ action: String(action), scene: fullScene.trim() });
    session.lastScene = fullScene.trim();
    session.summary = await updateSummary({
      oldSummary: session.summary,
      recent: session.recent,
      state: session.state,
    });
  } catch (e) {
    console.error("Audio-to-Story Error:", e);
    sseWrite(res, {
      type: "error",
      message: e.message || "Local transcription failed",
    });
  } finally {
    res.end();
  }
});

// Direct text to story endpoint (for browser-based STT) with TTS
app.post("/text-to-story", async (req, res) => {
  try {
    const { sessionId, character, action } = req.body || {};

    if (!sessionId || !character || !action) {
      return res
        .status(400)
        .json({ error: "Missing sessionId, character or action" });
    }

    // SSE headers
    res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
    res.setHeader("Cache-Control", "no-cache, no-transform");
    res.setHeader("Connection", "keep-alive");

    sseWrite(res, { type: "status", message: "Generating story..." });

    const session = getOrCreateSession(sessionId);
    const prompt = buildPrompt({
      character,
      action,
      summary: session.summary,
      lastScene: session.lastScene,
      recent: session.recent,
    });

    let fullScene = "";
    let ttsBuffer = ""; // Buffer for TTS processing

    const stream = await client.responses.create({
      model: MODEL,
      input: prompt,
      stream: true,
    });

    for await (const event of stream) {
      if (event.type === "response.output_text.delta") {
        fullScene += event.delta;
        ttsBuffer += event.delta;

        // Send text delta for immediate display
        sseWrite(res, { type: "delta", text: event.delta });

        // Check if we have complete sentences for TTS
        const sentences = extractCompleteSentences(ttsBuffer);
        if (sentences.complete) {
          // Generate TTS audio for complete sentences
          try {
            const ttsResponse = await client.audio.speech.create({
              model: "tts-1",
              voice: "alloy",
              input: sentences.complete,
              response_format: "mp3",
            });

            const audioBuffer = await ttsResponse.arrayBuffer();
            const audioBase64 = Buffer.from(audioBuffer).toString("base64");

            // Send audio chunk to client
            sseWrite(res, {
              type: "audio",
              audio: audioBase64,
              format: "mp3",
            });
          } catch (ttsError) {
            console.error("TTS Error:", ttsError);
            // Continue without TTS if it fails
          }

          ttsBuffer = sentences.remaining;
        }
      } else if (event.type === "response.error") {
        sseWrite(res, {
          type: "error",
          message: event.error?.message || "OpenAI error",
        });
      } else if (event.type === "response.completed") {
        // Process any remaining text for TTS
        if (ttsBuffer.trim()) {
          try {
            const ttsResponse = await client.audio.speech.create({
              model: "tts-1",
              voice: "alloy",
              input: ttsBuffer.trim(),
              response_format: "mp3",
            });

            const audioBuffer = await ttsResponse.arrayBuffer();
            const audioBase64 = Buffer.from(audioBuffer).toString("base64");

            sseWrite(res, {
              type: "audio",
              audio: audioBase64,
              format: "mp3",
            });
          } catch (ttsError) {
            console.error("TTS Error for remaining text:", ttsError);
          }
        }

        sseWrite(res, { type: "done" });
      }
    }

    // Update session memory
    session.recent.push({ action: String(action), scene: fullScene.trim() });
    session.lastScene = fullScene.trim();
    session.summary = await updateSummary({
      oldSummary: session.summary,
      recent: session.recent,
      state: session.state,
    });
  } catch (e) {
    console.error("Text-to-Story Error:", e);
    sseWrite(res, { type: "error", message: e.message || "Processing failed" });
  } finally {
    res.end();
  }
});

function sseWrite(res, data) {
  res.write(`data: ${JSON.stringify(data)}\n\n`);
}

// Helper function to extract complete sentences for TTS
function extractCompleteSentences(text) {
  const sentences = text.split(/(?<=[\.!\?])\s+/);
  if (sentences.length <= 1) {
    return { complete: "", remaining: text };
  }

  const lastSentence = sentences.pop();
  const completeSentences = sentences.join(" ");

  return {
    complete: completeSentences,
    remaining: lastSentence,
  };
}

// body:
// { sessionId, character, action: "A"|"B"|string }
app.post("/stream", async (req, res) => {
  const { sessionId, character, action } = req.body || {};
  if (!sessionId || !character || !action) {
    return res
      .status(400)
      .json({ error: "Missing sessionId, character or action" });
  }

  const session = getOrCreateSession(sessionId);

  // Build prompt with summary + lastScene + recent
  const prompt = buildPrompt({
    character,
    action,
    summary: session.summary,
    lastScene: session.lastScene,
    recent: session.recent,
  });

  // SSE headers
  res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  sseWrite(res, { type: "open" });

  let fullScene = ""; // accumulate streamed text to update memory later

  try {
    const stream = await client.responses.create({
      model: MODEL,
      input: prompt,
      stream: true,
    });

    for await (const event of stream) {
      if (event.type === "response.output_text.delta") {
        fullScene += event.delta;
        sseWrite(res, { type: "delta", text: event.delta });
      } else if (event.type === "response.error") {
        sseWrite(res, {
          type: "error",
          message: event.error?.message || "OpenAI error",
        });
      } else if (event.type === "response.completed") {
        sseWrite(res, { type: "done" });
      }
    }
  } catch (err) {
    sseWrite(res, {
      type: "error",
      message: err?.message || "Upstream failure",
    });
  } finally {
    res.end();
  }

  // --- Memory maintenance (after sending response) ---
  try {
    session.recent.push({ action: String(action), scene: fullScene.trim() });

    // Always remember the lastScene (for A/B resolution next turn)
    session.lastScene = fullScene.trim();
    session.summary = await updateSummary({
      oldSummary: session.summary,
      recent: session.recent,
      state: session.state,
    });
  } catch {
    // If summarization fails, keep current memory as-is
  } finally {
    console.log(`${session.summary}\n`);
    // console.log(`${JSON.stringify(session.recent)}\n`);
    console.log(`${JSON.stringify(session.state)}\n`);
  }
});

function buildPrompt({ character, action, summary, lastScene, recent }) {
  const isAB = typeof action === "string" && /^[ab]$/i.test(action);
  const normalized = isAB ? action.toUpperCase() : action;

  return `
You are an immersive fantasy storyteller.

Global rules:
- 2nd person ("you...").
- Tight narration (max 60 words).
- Maintain continuity using the summary. Do not contradict facts.
- End with exactly:
  A) <option A>
  B) <option B>
  (Or describe your own custom action.)

=== CANON SUMMARY (compact memory of prior story) ===
${summary || "(none)"}

=== RECENT EXCHANGES (most recent first) ===
${
  [...recent]
    .reverse()
    .map((r, i) => `#${i + 1} Player: ${r.action}\nScene: ${r.scene}`)
    .join("\n\n") || "(none)"
}

=== CHARACTER ===
${JSON.stringify(character, null, 2)}

${
  isAB
    ? `=== LAST SCENE (with options A/B) ===
${lastScene}

=== PLAYER CHOICE ===
The player chose option "${normalized}". Continue accordingly.`
    : `=== PLAYER CUSTOM ACTION ===
${normalized}
Continue the story treating this as the player's intent.`
}

=== OUTPUT FORMAT (render exactly like this) ===
<Narration text...>
A) <New option A>
B) <New option B>
(Or describe your own custom action.)
`.trim();
}

// Audio-only streaming endpoint (processes full text to audio)
app.post("/audio-stream", async (req, res) => {
  try {
    const { sessionId, text } = req.body || {};

    if (!sessionId || !text) {
      return res.status(400).json({ error: "Missing sessionId or text" });
    }

    // SSE headers
    res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
    res.setHeader("Cache-Control", "no-cache, no-transform");
    res.setHeader("Connection", "keep-alive");

    sseWrite(res, { type: "status", message: "Converting to audio..." });

    // Process the full text for TTS
    const sentences = extractCompleteSentences(text + "."); // Ensure we get all sentences
    const allSentences = sentences.complete;

    if (allSentences.trim()) {
      try {
        console.log(`Converting to audio: "${allSentences.slice(0, 50)}..."`);

        const ttsResponse = await client.audio.speech.create({
          model: "tts-1",
          voice: "alloy",
          input: allSentences,
          response_format: "mp3",
        });

        // Convert to base64 and stream
        const audioBuffer = Buffer.from(await ttsResponse.arrayBuffer());
        const base64Audio = audioBuffer.toString("base64");

        sseWrite(res, {
          type: "audio",
          audio: base64Audio,
          format: "mp3",
        });

        console.log(`Audio generated and sent (${audioBuffer.length} bytes)`);
      } catch (ttsError) {
        console.error("TTS Error:", ttsError);
        sseWrite(res, { type: "error", message: "Audio generation failed" });
      }
    }

    sseWrite(res, { type: "done" });
    res.end();
  } catch (e) {
    console.error("Audio Stream Error:", e);
    sseWrite(res, { type: "error", message: "Error generating audio" });
    res.end();
  }
});

// Text-only streaming endpoint (no audio processing)
app.post("/text-stream", async (req, res) => {
  try {
    const { sessionId, character, action } = req.body || {};

    if (!sessionId || !character || !action) {
      return res
        .status(400)
        .json({ error: "Missing sessionId, character or action" });
    }

    // SSE headers
    res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
    res.setHeader("Cache-Control", "no-cache, no-transform");
    res.setHeader("Connection", "keep-alive");

    sseWrite(res, { type: "status", message: "Generating story..." });

    const session = getOrCreateSession(sessionId);
    const prompt = buildPrompt({
      character,
      action,
      summary: session.summary,
      lastScene: session.lastScene,
      recent: session.recent,
    });

    let fullScene = "";

    const stream = await client.responses.create({
      model: MODEL,
      input: prompt,
      stream: true,
    });

    for await (const event of stream) {
      if (event.type === "response.output_text.delta") {
        const textDelta = event.delta || "";
        fullScene += textDelta;

        // Send only text delta (no audio processing)
        sseWrite(res, { type: "delta", text: textDelta });
      }
    }

    // Update session with the new scene
    session.lastScene = fullScene;
    session.recent.push({ action, scene: fullScene });
    if (session.recent.length > 3) session.recent.shift();

    // Update summary periodically
    if (session.recent.length >= 3) {
      session.summary = await updateSummary(session.summary, session.recent);
      session.recent = [];
    }

    sseWrite(res, { type: "done" });
    res.end();
  } catch (e) {
    console.error("Text Stream Error:", e);
    sseWrite(res, { type: "error", message: "Error generating story" });
    res.end();
  }
});

const port = Number(process.env.PORT || 8787);
app.listen(port, () => {
  console.log(`MVP server on http://localhost:${port}`);
});
