// sessions.js
export const sessions = new Map(); // sessionId -> Session

export function getOrCreateSession(sessionId) {
  let s = sessions.get(sessionId);
  if (!s) {
    s = {
      summary: "", // compressed memory of everything so far
      lastScene: "", // full text of last streamed scene (with A/B)
      recent: [], // last N exchanges: { action, scene }
      state: {
        // optional structured state you can grow over time
        location: null,
        flags: {},
        inventory: [],
      },
    };
    sessions.set(sessionId, s);
  }
  return s;
}
