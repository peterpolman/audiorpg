import "dotenv/config";
import express from "express";
import OpenAI from "openai";
import { getOrCreateSession } from "../sessions.js";
import { updateSummary } from "../summarizer.js";

const app = express();
app.use(express.json());

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const MODEL = process.env.MODEL || "gpt-4o-mini";

function extractCompleteSentences(text) {
  const sentences = [];
  const sentenceRegex = /[.!?]+[\s]*(?=[A-Z]|$)/g;
  let lastIndex = 0;
  let match;

  while ((match = sentenceRegex.exec(text)) !== null) {
    const sentence = text.slice(lastIndex, match.index + match[0].length).trim();
    if (sentence) {
      sentences.push(sentence);
    }
    lastIndex = match.index + match[0].length;
  }

  const remaining = text.slice(lastIndex).trim();
  
  return { sentences, remaining };
}

function buildPrompt(userInput, session) {
  let prompt = `You are an immersive RPG narrator creating an engaging fantasy adventure. The player has said: "${userInput}"\n\n`;
  
  if (session.summary) {
    prompt += `Story Summary: ${session.summary}\n\n`;
  }
  
  if (session.conversationHistory.length > 0) {
    const recentHistory = session.conversationHistory.slice(-10);
    prompt += "Recent conversation:\n";
    recentHistory.forEach(item => {
      prompt += `Player: ${item.user}\n`;
      prompt += `Narrator: ${item.assistant}\n`;
    });
    prompt += "\n";
  }
  
  prompt += `Continue the story based on the player's action. Describe what happens next in an engaging, immersive way. Keep responses focused and under 150 words.`;
  
  return prompt;
}

// Text streaming endpoint
app.get("/text-stream", async (req, res) => {
  const userInput = req.query.message;
  
  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Access-Control-Allow-Origin": "*",
  });

  try {
    const session = getOrCreateSession();
    const prompt = buildPrompt(userInput, session);

    const stream = await client.chat.completions.create({
      model: MODEL,
      messages: [{ role: "user", content: prompt }],
      stream: true,
      max_tokens: 300,
      temperature: 0.8,
    });

    let fullResponse = "";
    let lastFullScene = "";
    
    for await (const chunk of stream) {
      const content = chunk.choices[0]?.delta?.content || "";
      if (content) {
        fullResponse += content;
        res.write(`data: ${JSON.stringify({ text: content })}\n\n`);
      }
    }

    session.conversationHistory.push({
      user: userInput,
      assistant: fullResponse,
    });

    if (session.conversationHistory.length % 5 === 0) {
      updateSummary(session);
    }

    res.write(`data: ${JSON.stringify({ done: true })}\n\n`);
  } catch (error) {
    console.error("Error in text stream:", error);
    res.write(`data: ${JSON.stringify({ error: error.message })}\n\n`);
  } finally {
    res.end();
  }
});

// Audio streaming endpoint
app.get("/audio-stream", async (req, res) => {
  const text = req.query.text;
  
  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Access-Control-Allow-Origin": "*",
  });

  try {
    if (!text) {
      res.write(`data: ${JSON.stringify({ error: "No text provided" })}\n\n`);
      return;
    }

    const { sentences } = extractCompleteSentences(text);
    
    for (const sentence of sentences) {
      try {
        const response = await client.audio.speech.create({
          model: "tts-1",
          voice: "alloy",
          input: sentence,
        });

        const arrayBuffer = await response.arrayBuffer();
        const base64 = Buffer.from(arrayBuffer).toString('base64');
        
        res.write(`data: ${JSON.stringify({ 
          audio: base64,
          sentence: sentence 
        })}\n\n`);
      } catch (error) {
        console.error(`Error generating audio for sentence: ${sentence}`, error);
        res.write(`data: ${JSON.stringify({ 
          error: `Audio generation failed for: ${sentence}` 
        })}\n\n`);
      }
    }

    res.write(`data: ${JSON.stringify({ done: true })}\n\n`);
  } catch (error) {
    console.error("Error in audio stream:", error);
    res.write(`data: ${JSON.stringify({ error: error.message })}\n\n`);
  } finally {
    res.end();
  }
});

// Health check endpoint
app.get("/health", (req, res) => {
  res.json({ status: "ok" });
});

// Export the Express app for Vercel
export default app;