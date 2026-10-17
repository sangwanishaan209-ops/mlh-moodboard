# Moodboard

A pastel, freeform moodboard with movable image and note cards, AI answers grounded in the whole board, and ElevenLabs dictation.

## Run locally

1. Create a virtual environment and install dependencies:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and add your API keys. The keys stay on the server and are ignored by Git.

3. Start the app:

   ```powershell
   uvicorn main:app --reload
   ```

Paste, drag, or upload images onto the canvas. Ask Board sends all note text and compressed copies of the board's images together as one context to OpenAI, and keeps recent chat turns available for follow-up questions. The sidebar microphone adds dictated notes to the same canvas; the Ask Board microphone transcribes questions with ElevenLabs Scribe. Audio is recorded in the browser and sent to the app server for transcription. Board cards themselves remain in this browser's IndexedDB.

Set `OPENAI_MODEL` in `.env` if you want to use a different model available to your OpenAI API account.
