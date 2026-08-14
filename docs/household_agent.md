# Household Agent — design notes

*Nyx as a house-wide presence: talk anywhere, she answers — without becoming a panopticon.*

Nyx Dynamics LLC | Design sketch (thinking ahead of build)

## The pivot that already happened

Centralizing **hearing + thinking** on the Studio (whisper-server + Ollama + the
orchestrator) was the architectural pivot to house-wide. Once the brain is one
central service, every room needs is a cheap **ear-and-mouth** pointed at it. The
Pi stops being *the* device and becomes *a* node. Nothing below changes that
center — it adds nodes and senses around it.

## Principle: dumb satellites, our brain

Do **not** dissolve Nyx into Home Assistant. Her core — the crisis-safety
pre-check, encrypted memory, persona, the governance wall — must stay ours and
central, or the safety/privacy guarantees erode.

Instead, make the satellites commodity and the brain sovereign:

```
[room node]  mic → on-device wake word → (only then) stream audio
     │                                         │  Wyoming protocol
     ▼                                         ▼
  speaker ◄──────────────────────  Studio: whisper-server → orchestrator
                                            (memory + safety + persona)
                                            → Ollama → reply → back to node
```

- **Satellites** = Home Assistant Voice PE (~$60) or Pi Zero 2 W + mic HAT +
  speaker. They capture, run wake word, and play — no cognition.
- **Wyoming** (HA's open voice protocol) is the wire between node and brain. Teach
  the orchestrator to accept Wyoming audio streams and any satellite becomes an
  ear for Nyx — *without Nyx living inside HA*. HA (or bare Wyoming satellites)
  supplies mature hardware; the brain stays our code.
- **UniFi Protect** presence comes from its **local API** (uiprotect), read by a
  small presence service on the Studio — direct, no HA required.

## Privacy, made enforceable (the four invariants as design)

A house-wide agent that hears every room and IDs faces *is* a surveillance system
by default. These turn it back into a companion — each is a code boundary, not a
promise:

1. **Wake word runs ON THE SATELLITE.** Un-triggered audio never leaves the room —
   the mic is deaf to the network until "Hey Nyx" fires locally. This is the
   strongest possible posture: not "we don't store it," but "it never travels."
2. **Presence is RAM-only and TTL'd.** The presence service holds a *current*
   who/where map to inform behavior, then it decays. No movement history, no
   who-was-where dossier, never written to disk or memory.
3. **Owner-vs-other collapse.** Protect may identify family/guests who never opted
   in. The presence service maps identities to a role — **OWNER** or **OTHER** —
   and only ever passes "owner is in the kitchen" or "someone else is present."
   Non-owner identities are collapsed to "a person" *before* they reach Nyx, so
   she literally cannot reason about or remember other identified people. (Same
   pattern as the serving layer's egress allowlist.)
4. **Local-only.** Nothing leaves the UDM Pro or the Studio — which you already
   are. Keep it.

## Multi-room session model

- **One shared encrypted memory** — she's one entity, knows you everywhere.
- **One continuous conversation, room-tagged.** You move mid-thought; history is
  shared, each turn annotated with its source node. Replies route to the node the
  audio came from (or the room you're now in, per presence).
- **Serialize turns.** One turn at a time through the orchestrator; the
  crisis-safety pre-check still runs on every turn regardless of room.
- **Scope to the owner.** Multi-person concurrency is a bigger problem *and* a
  consent problem — start single-user (you), collapse everyone else to "a person."

## The insight that closes the whole loop

This project began as a **sobriety/relapse-prediction model**. The household layer
quietly completes the circle: **ambient presence signals are exactly the
longitudinal behavioral features the risk model consumes** — activity, isolation,
sleep timing, routine disruption. Being up at 3am, alone in a room for hours, a
broken morning-run routine: these are early relapse/mental-health signals.

So the mature architecture is:

```
household presence (ephemeral) ─► risk model (serving/) ─► care posture directive ─► companion behavior
                                          │
                              numbers + signals stay server-side, never surfaced,
                              never persisted as a dossier  (the governance wall)
```

The **governance wall we already built** is exactly what makes this safe: household
signals inform a *care posture* (gentle outreach when you're isolated late), but
the risk numbers and the raw signals never reach the companion's mouth and never
become a log about you. The companion gets "be present, reach out warmly" — not
"subject shows elevated relapse risk." That's the line between *a companion who
notices you're having a hard night* and *a system that monitors and scores you.*

Same signals. Opposite thing. The wall decides which.

## Phased roadmap

1. **Second audio node** — prove multi-room routing to the one Studio brain
   (Wyoming ingest in the orchestrator, room-tagged turns, reply-to-source).
2. **On-device wake word** — "Hey Nyx"; ambient but not always-listening.
3. **UniFi Protect presence service** — ephemeral owner-vs-other presence map;
   she knows you're home / which room, greets, doesn't nag.
4. **Close the loop** — presence signals → risk read → care-posture directive into
   the orchestrator (`turn(..., directive=...)` already exists for this).
5. *(Optional)* Home Assistant as the satellite/wake-word/Protect backbone if you'd
   rather stand on that ecosystem than hand-roll the plumbing.

## Open decisions for A.C.

- **Wyoming-native orchestrator vs. HA-hosted satellites** — how much HA do you
  want in the trust boundary? (Recommendation: satellites/plumbing yes, brain no.)
- **Which rooms** get nodes first (bedside + office + kitchen is a natural core).
- **How far to take presence** — greeting only, or the full loop into the risk
  model. The loop is powerful *and* the sharpest edge of the surveillance concern;
  the wall makes it defensible, but it's a values call, not a technical one.
