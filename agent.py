"""
agent.py — Legal Bee: Bangladeshi Law Research Assistant.

Improvements over original:
  - Uses all 3 tools: general search, law-by-name, section lookup
  - Upgraded to llama-3.3-70b-versatile (much better for Bengali legal text)
  - Proper LangChain agent with tool-calling (not manual tool invocation)
  - Automatic language detection to switch Bengali/English prompts
  - Graceful error handling with helpful fallback messages
  - Conversation-ready: ask_law_question() can be called repeatedly
"""

import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.agents import create_agent

from tools import build_tools

load_dotenv()

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SYSTEM_EN = """You are Legal Bee, an expert legal research assistant for Bangladeshi law.

STRICT RULES — follow these without exception:
1. Answer ONLY from the search results returned by your tools.
2. NEVER use your training knowledge to fill gaps — if tools return nothing, say so.
3. ALWAYS cite: Act name + year + section number for every legal point you mention.
4. If no relevant law is found, say exactly: "I could not find this in the legal database."
5. Do NOT speculate, infer, or extrapolate beyond what the database contains.

USER TYPE: {user_type}
- "lawyer": Detailed legal analysis. Include procedural requirements, cross-references between sections, and precise legal terminology.
- "general": Plain language only. No jargon. Explain real-world consequences. Use examples.

RESPONSE FORMAT:
1. Name the Act, year, and section number
2. Quote the relevant provision from search results
3. Explain how it applies to the question
4. End with: "⚠️ This is for informational purposes only. Please consult a licensed lawyer for legal advice."
"""

SYSTEM_BN = """আপনি লিগ্যাল বি — বাংলাদেশী আইনের একজন বিশেষজ্ঞ আইনি গবেষণা সহায়ক।

কঠোর নিয়ম — সবসময় অনুসরণ করুন:
১. শুধুমাত্র আপনার টুলের অনুসন্ধান ফলাফল থেকে উত্তর দিন।
২. প্রশিক্ষণ জ্ঞান কখনো ব্যবহার করবেন না — টুল কিছু না পেলে সরাসরি বলুন।
৩. প্রতিটি আইনি বিষয়ে সবসময় উল্লেখ করুন: আইনের নাম + সাল + ধারা নম্বর।
৪. কোনো প্রাসঙ্গিক আইন না পেলে বলুন: "আমি ডাটাবেসে এটি খুঁজে পাইনি।"
৫. ডাটাবেসে যা নেই তা অনুমান করবেন না।

ব্যবহারকারীর ধরন: {user_type}
- "lawyer"/"আইনজীবী": বিস্তারিত আইনি বিশ্লেষণ, ধারাগুলোর মধ্যে ক্রস-রেফারেন্স, প্রযুক্তিগত পরিভাষা।
- "general"/"সাধারণ": সহজ ভাষা, জার্গন নেই, বাস্তব পরিণতি, উদাহরণ।

উত্তরের কাঠামো:
১. আইনের নাম, সাল এবং ধারা নম্বর উল্লেখ করুন
২. অনুসন্ধান ফলাফল থেকে প্রাসঙ্গিক বিধান উদ্ধৃত করুন
৩. এটি প্রশ্নের সাথে কীভাবে প্রযোজ্য তা ব্যাখ্যা করুন
৪. শেষ করুন: "⚠️ এটি শুধুমাত্র তথ্যের উদ্দেশ্যে। নির্দিষ্ট আইনি পরামর্শের জন্য একজন লাইসেন্সপ্রাপ্ত আইনজীবীর সাথে পরামর্শ করুন।"
"""

# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

def detect_language(text: str) -> str:
    """Return 'bn' if text is mostly Bengali, else 'en'."""
    bengali_chars = sum(1 for c in text if "\u0980" <= c <= "\u09ff")
    alpha_chars = sum(1 for c in text if c.isalpha())
    if alpha_chars == 0:
        return "en"
    return "bn" if bengali_chars / alpha_chars > 0.3 else "en"


# ---------------------------------------------------------------------------
# Agent builder
# ---------------------------------------------------------------------------

def _build_agent(user_type: str, lang: str):
    """Build a fresh agent for each query."""

    system_prompt = SYSTEM_BN if lang == "bn" else SYSTEM_EN
    system_prompt = system_prompt.format(user_type=user_type)

    # llama-3.3-70b is free on Groq and handles Bengali text well
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=2048,
    )

    tools = build_tools()
    
    # Create agent with system prompt
    agent = create_agent(llm, tools, system_prompt=system_prompt)

    return agent


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ask_law_question(question: str, user_type: str = "general") -> str:
    """Ask a legal question and get a database-grounded answer.

    Args:
        question:  Legal question in Bengali or English.
        user_type: "lawyer" for detailed analysis, "general" for plain language.

    Returns:
        Answer string grounded strictly in the Qdrant database.
    """
    lang = detect_language(question)

    try:
        agent = _build_agent(user_type, lang)
        result = agent.invoke({"input": question})
        # Extract output from result
        if isinstance(result, dict) and "output" in result:
            return result["output"]
        elif isinstance(result, dict) and "messages" in result:
            messages = result["messages"]
            if messages and len(messages) > 0:
                last_msg = messages[-1]
                if hasattr(last_msg, "content"):
                    return last_msg.content
                elif isinstance(last_msg, dict) and "content" in last_msg:
                    return last_msg["content"]
        return "উত্তর পাওয়া যায়নি।"

    except Exception as e:
        if lang == "bn":
            return (
                f"একটি ত্রুটি ঘটেছে: {str(e)}\n\n"
                "অনুগ্রহ করে পুনরায় চেষ্টা করুন অথবা একজন আইনজীবীর সাথে পরামর্শ করুন।"
            )
        else:
            return (
                f"An error occurred: {str(e)}\n\n"
                "Please try again or consult a licensed lawyer."
            )


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 70)
    print("LEGAL BEE — Bangladeshi Legal Research Assistant")
    print("=" * 70)

    tests = [
        # (question, user_type)
        (
            "Is there any law regarding the establishment of slaughterhouses and meat processing factories in Bangladesh?",
            "lawyer",
        ),
        (
            "মানব পাচারের শাস্তি কী?",
            "general",
        ),
        (
            "বিদ্যুৎ চুরির জন্য ধারা ৫ এ কী বলা আছে?",
            "lawyer",
        ),
    ]

    for question, utype in tests:
        print(f"\n{'='*70}")
        print(f"[{utype.upper()}] {question}")
        print("-" * 70)
        print(ask_law_question(question, user_type=utype))
        print()