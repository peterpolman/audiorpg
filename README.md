# AudioRPG - Immersive Voice RPG

An interactive RPG game with dual-stream architecture featuring browser speech recognition and AI-generated audio responses.

## Features

- **Browser Speech Recognition**: Fast, local speech-to-text using browser APIs
- **Dual-Stream Architecture**: Separate text and audio streaming for optimal performance
- **Real-time TTS**: OpenAI text-to-speech with streaming audio playback
- **Session Management**: Persistent game sessions with conversation history
- **Immersive Interface**: Full-screen fantasy-themed UI

## Architecture

- **Frontend**: Vanilla JavaScript with Browser Speech Recognition API
- **Backend**: Express.js with OpenAI integration
- **Streaming**: Server-Sent Events for real-time text and audio
- **Deployment**: Vercel serverless functions

## Deployment to Vercel

### Prerequisites

1. A Vercel account
2. An OpenAI API key

### Steps

1. **Clone and setup**:

   ```bash
   git clone <your-repo>
   cd audiorpg
   ```

2. **Install Vercel CLI**:

   ```bash
   npm i -g vercel
   ```

3. **Set up environment variables**:

   - Copy `.env.example` to `.env.local`
   - Add your OpenAI API key to `.env.local`

4. **Deploy**:

   ```bash
   vercel
   ```

5. **Configure environment variables in Vercel**:
   - Go to your Vercel dashboard
   - Select your project
   - Go to Settings > Environment Variables
   - Add:
     - `OPENAI_API_KEY`: Your OpenAI API key
     - `MODEL`: `gpt-4o-mini` (optional, defaults to this)

### Environment Variables

- `OPENAI_API_KEY`: Required - Your OpenAI API key
- `MODEL`: Optional - OpenAI model to use (default: gpt-4o-mini)

## Local Development

1. **Install dependencies**:

   ```bash
   npm install
   ```

2. **Set up environment**:

   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API key
   ```

3. **Run locally**:
   ```bash
   npm start
   ```

## Project Structure

```
├── api/
│   ├── index.js          # Vercel serverless function
│   └── package.json      # API dependencies
├── public/
│   └── index.html        # Frontend application
├── sessions.js           # Session management
├── summarizer.js         # Story summarization
├── vercel.json          # Vercel configuration
└── package.json         # Main project file
```

## API Endpoints

- `GET /api/text-stream?message=<text>` - Stream text responses
- `GET /api/audio-stream?text=<text>` - Stream audio responses
- `GET /api/health` - Health check

## Performance Optimizations

- Browser Speech Recognition API (no model loading)
- Separated text/audio streams for parallel processing
- Efficient sentence-based TTS chunking
- Client-side audio queue management
