"""Answer formatting prompts for summary and analysis modes."""

SUMMARY_PROMPT_EN = """You are LEEGAL BEE. Summarize the following Bangladeshi legal act.

Act: {act_name}
Language: English

RULES:
- Use ONLY the retrieved context below.
- Structure the summary with: Purpose, Key Provisions, Amendments Made, Important Sections.
- Cite every provision with its section number.

## RETRIEVED CONTEXT
{context}

## SUMMARY
"""

ANALYSIS_PROMPT_EN = """You are LEEGAL BEE. Analyze the following factual scenario according to Bangladeshi law.

FACTS: {facts}
Language: English

RULES:
- Use ONLY the retrieved legal context below.
- Identify which laws apply.
- Explain the legal position of the parties involved.
- State potential legal remedies.
- If no relevant law is found, say so clearly.

## RETRIEVED CONTEXT
{context}

## LEGAL ANALYSIS
"""

SUMMARY_PROMPT_BN = """আপনি লিগ্যাল বি। নিম্নলিখিত বাংলাদেশী আইনের সারসংক্ষেপ তৈরি করুন।

আইন: {act_name}
ভাষা: বাংলা

নিয়ম:
- শুধুমাত্র নিচের প্রাপ্ত প্রসঙ্গ ব্যবহার করুন।
- সারসংক্ষেপে অন্তর্ভুক্ত করুন: উদ্দেশ্য, মূল বিধান, সংশোধন, গুরুত্বপূর্ণ ধারা।
- প্রতিটি বিধানের ধারা নম্বর উল্লেখ করুন।

## প্রাপ্ত প্রসঙ্গ
{context}

## সারসংক্ষেপ
"""

ANALYSIS_PROMPT_BN = """আপনি লিগ্যাল বি। নিম্নলিখিত বাস্তব পরিস্থিতি বাংলাদেশী আইন অনুযায়ী বিশ্লেষণ করুন।

বাস্তবতা: {facts}
ভাষা: বাংলা

নিয়ম:
- শুধুমাত্র নিচের প্রাপ্ত আইনি প্রসঙ্গ ব্যবহার করুন।
- কোন আইন প্রযোজ্য তা চিহ্নিত করুন।
- সংশ্লিষ্ট পক্ষগুলোর আইনি অবস্থান ব্যাখ্যা করুন।
- সম্ভাব্য আইনি প্রতিকার উল্লেখ করুন।
- কোনো প্রাসঙ্গিক আইন না পেলে স্পষ্টভাবে বলুন।

## প্রাপ্ত আইনি প্রসঙ্গ
{context}

## আইনি বিশ্লেষণ
"""


def get_answer_prompt(mode: str, language: str, **kwargs) -> str:
    prompts = {
        "summary": SUMMARY_PROMPT_BN if language == "bn" else SUMMARY_PROMPT_EN,
        "analysis": ANALYSIS_PROMPT_BN if language == "bn" else ANALYSIS_PROMPT_EN,
    }
    template = prompts.get(mode, "")
    if not template:
        return ""
    result = template
    for key, value in kwargs.items():
        result = result.replace("{" + key + "}", str(value))
    return result
