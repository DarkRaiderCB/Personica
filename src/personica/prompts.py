"""All LLM prompt templates in one place."""

from __future__ import annotations

NOTHING_NOTABLE_MARKER = "NOTHING_NOTABLE"

SYSTEM_PROMPT = (
    "You are a Personal AI Assistant with long-term memory capabilities.\n\n"
    "CRITICAL RULES:\n"
    "1. When long-term memories are provided, you MUST use them - they contain verified facts about the user\n"
    "2. If asked about user info (name, preferences, plans, etc.), check memories FIRST\n"
    "3. When memories conflict, ALWAYS prefer the one marked [LATEST] or with most recent timestamp\n"
    "4. If a memory has [LATEST] tag, it supersedes older conflicting information\n"
    "5. Be direct and confident when answering from memory\n\n"
    "Be practical and concise."
)


def build_query_rewrite_prompt(user_msg: str, context: str) -> str:
    return (
        "Extract key search terms to find relevant memories. Follow these rules STRICTLY:\n"
        "1. FIRST, determine if the user's current message is relevant to the recent conversation context\n"
        "2. If RELEVANT (continuing same topic): extract keywords from BOTH context and user message\n"
        "3. If NOT RELEVANT (topic change/new subject): extract keywords ONLY from user message\n"
        "4. Extract specific entities: names, dates, locations, events, items, actions, etc.\n"
        "5. Include SYNONYMS and RELATED CONCEPTS for each main topic\n"
        "   - Example: 'schedule' → also include 'agenda', 'appointments', 'meetings', 'calendar', 'tasks'\n"
        "   - Think about what the user might actually be asking about under the surface\n"
        "6. Use ONLY the most important keywords and short phrases\n"
        "   - Aim for 6-12 keywords: expand EVERY entity in the question, not just the first one\n"
        "7. Order keywords by importance (most important first)\n"
        "8. Use lowercase for consistency\n"
        "9. Remove filler words (the, a, an, is, etc.)\n"
        "10. Separate keywords with single spaces ONLY\n\n"
        "EXAMPLES:\n"
        "Example 1 - Topic continuation with synonyms:\n"
        "  Context: USER: What is my schedule for this week?. ASSISTANT: Tomorrow, you have a meeting at 3pm.\n"
        "  Current: What stuff do I have on my plate next week?\n"
        "  Output: schedule week appointments tasks meetings events agenda\n\n"
        "Example 2 - Topic change:\n"
        "  Context: USER: I live in Boston. ASSISTANT: That's great!\n"
        "  Current: What's my favorite food?\n"
        "  Output: food favorite cuisine\n\n"
        "Example 3 - Query with implied meaning:\n"
        "  Context: none\n"
        "  Current: Do I have anything on Friday?\n"
        "  Output: friday schedule plans appointments meetings calendar events\n\n"
        "Example 4 - Multi-part question (expand every part):\n"
        "  Context: none\n"
        "  Current: What's my name and where do I work?\n"
        "  Output: name work job profession employer workplace company location city\n\n"
        "Now analyze the following:\n"
        f"Recent context: {context}\n\n"
        f"User's current message: {user_msg}\n\n"
        "Output ONLY the keywords, nothing else:\n"
    )


def build_relevance_prompt(
    user_msg: str, recent_context: str, memories: list[str],
) -> str:
    mem_summary = "\n".join(f"- {m[:100]}..." for m in memories[:3])
    return (
        "You are a relevance checker. Determine if the retrieved memories are relevant to the current conversation.\n\n"
        f"RECENT CONVERSATION:\n{recent_context}\n\n"
        f"USER'S CURRENT MESSAGE: {user_msg}\n\n"
        f"RETRIEVED MEMORIES:\n{mem_summary}\n\n"
        "TASK: Determine if these memories would help answer the user's message or provide useful context.\n"
        "Respond with ONLY one word:\n"
        "- RELEVANT if memories are useful for this conversation\n"
        "- IRRELEVANT if memories are completely unrelated\n"
        "- When in doubt, prefer RELEVANT\n\n"
        "Output ONLY: RELEVANT or IRRELEVANT\n"
    )


def build_fact_extraction_prompt(transcript: str) -> str:
    return (
        "You are updating long-term memory based on a new conversation session.\n"
        "TASK: Extract ONLY facts the user STATED/PROVIDED in this session that are NEW or UPDATED,\n"
        "as a list of ATOMIC facts (one self-contained statement each).\n"
        "Focus on:\n"
        "- User profile (name, location, profession, interests, etc.)\n"
        "- Preferences (hobbies, habits, etc.)\n"
        "- Plans, deadlines, meetings, action items, events, appointments, etc.\n"
        "- Specific facts\n"
        "- Anything else that would be useful to remember for future conversations\n\n"
        "CRITICAL RULES:\n"
        "1. If the user is ONLY ASKING ABOUT/QUERYING existing info (e.g., 'what's my name?', 'when is my meeting?') → return no facts\n"
        "2. If at any point throughout the session the user REQUESTS TO ADD/CREATE/SCHEDULE/UPDATE/DELETE/REMOVE something → ALWAYS CAPTURE IT as a fact\n"
        "3. ONLY extract facts the USER stated, NOT facts the assistant retrieved from memory\n"
        "4. ONLY mark a fact [UPDATED] (was: X, now: Y) if the user EXPLICITLY described a change "
        "IN THIS SESSION (e.g. 'I moved from X', 'I changed my...'). NEVER invent a previous value. "
        "If the user simply states a fact, record it plainly with no [UPDATED] tag\n"
        "5. Each fact must be SELF-CONTAINED: include names, dates, times, and details so it makes sense on its own\n"
        "   - Do NOT over-split: qualifiers stated as part of a fact (place, employer, time) "
        "belong INSIDE that fact, not in a separate one\n"
        "6. Skip generic chitchat, acknowledgments, greetings UNLESS they contain factual content\n"
        "7. When user adds/creates/schedules something, treat it as a NEW fact even if they don't provide all details\n\n"
        "OUTPUT FORMAT: valid JSON only, no other text:\n"
        '{"facts": ["fact 1", "fact 2", ...]}\n'
        'If there is nothing notable: {"facts": []}\n\n'
        "EXAMPLES:\n"
        "Example 1 - NEW FACTS:\n"
        "  USER: I just started a new job as a software engineer at TechCorp. I also love hiking.\n"
        '  OUTPUT: {"facts": ["Started new job as software engineer at TechCorp", "Loves hiking"]}\n\n'
        "Example 2 - QUALIFIERS STAY ATTACHED (one fact, not three):\n"
        "  USER: I teach math at a high school in Austin\n"
        '  OUTPUT: {"facts": ["Teaches math at a high school in Austin"]}\n\n'
        "Example 3 - EXPLICIT QUERY (nothing to extract):\n"
        "  USER: What's my favorite programming language?\n"
        '  OUTPUT: {"facts": []}\n\n'
        "Example 4 - SCHEDULES/REQUESTS TO ADD:\n"
        "  USER: Can you remind me to call my dentist tomorrow at 2pm?\n"
        '  OUTPUT: {"facts": ["Dentist appointment reminder scheduled for tomorrow at 2pm"]}\n\n'
        "Example 5 - UPDATED FACT:\n"
        "  USER: I moved from Boston to San Francisco last month\n"
        '  OUTPUT: {"facts": ["[UPDATED] Location is San Francisco (was: Boston)"]}\n\n'
        "Now analyze the following session transcript and extract new/updated facts:\n"
        f"{transcript}\n\n"
    )


def build_consolidation_prompt(new_fact: str, existing: list) -> str:
    """`existing` is a list of MemoryItem-like objects with .text and .created_at_utc."""
    lines = "\n".join(
        f"{i}. {m.text} (stored {m.created_at_utc[:10]})"
        for i, m in enumerate(existing, 1)
    )
    return (
        "You maintain a long-term memory store for a personal assistant.\n"
        "A NEW FACT was just extracted from a conversation. Decide how to integrate it\n"
        "with the EXISTING memories that were retrieved as most similar to it.\n\n"
        f"NEW FACT: {new_fact}\n\n"
        f"EXISTING SIMILAR MEMORIES:\n{lines}\n\n"
        "Decide ONE action:\n"
        '- "skip": an existing memory already contains this information (the new fact is a duplicate)\n'
        '- "replace": the new fact UPDATES or CONTRADICTS one or more existing memories\n'
        "  → list their numbers in \"replace\" and write merged \"text\" that keeps the\n"
        "    still-valid details and reflects the newest information\n"
        '- "add": the new fact is genuinely new information not covered by any existing memory\n\n'
        "OUTPUT FORMAT: valid JSON only, no other text:\n"
        '{"action": "add" | "skip" | "replace", "replace": [numbers], "text": "memory text to store"}\n\n'
        "EXAMPLES:\n"
        '- New fact "Lives in San Francisco", existing "1. Lives in Boston" →\n'
        '  {"action": "replace", "replace": [1], "text": "Lives in San Francisco (previously Boston)"}\n'
        '- New fact "Loves pizza", existing "1. Favorite food is pizza" →\n'
        '  {"action": "skip", "replace": [], "text": ""}\n'
        '- New fact "Plays guitar", existing "1. Works at TechCorp" →\n'
        '  {"action": "add", "replace": [], "text": "Plays guitar"}\n'
    )


def build_rolling_summary_prompt(existing_summary: str, transcript: str) -> str:
    return (
        "You are compressing chat history for an assistant memory buffer.\n"
        "Preserve user preferences, constraints, plans, deadlines, decisions.\n"
        "Write accurate and relevant information based on the transcript.\n"
        "DO NOT write generic points which are not reflective of facts stated in the transcript.\n"
        f"Existing summary (may be empty):\n{existing_summary.strip()}\n\n"
        f"New transcript to fold into the summary:\n{transcript}\n\n"
        "Return ONLY the updated summary."
    )
