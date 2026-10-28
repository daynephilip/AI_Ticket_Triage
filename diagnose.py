"""Check what LM Studio actually returns for a draft-style request.
Run:  python diagnose.py"""
import time

import triage

name, model = triage.PROVIDERS[0]
print(f"Testing {name}: model={triage.LMSTUDIO_MODEL}  url={triage.LMSTUDIO_URL}  "
      f"timeout={model.request_timeout}s\n")

system = "You are a support agent. Reply helpfully in numbered steps."
user = "I use a Mac and want to install Firefox. I have no version preference. How do I do it?"

start = time.time()
try:
    resp = model.invoke([("system", system), ("user", user)])
except Exception as e:
    print(f"CALL FAILED after {time.time() - start:.0f}s: {type(e).__name__}: {e}\n")
    print("-> Wrong model name, server not running, or timeout. Check LMSTUDIO_MODEL in .env,")
    print("   the Developer tab (Status: Running), and try raising LMSTUDIO_TIMEOUT.")
    raise SystemExit(1)

elapsed = time.time() - start
raw = str(resp.content)
cleaned = triage.clean(raw)
meta = getattr(resp, "response_metadata", {}) or {}
print(f"Time taken:     {elapsed:.0f}s")
print(f"finish_reason:  {meta.get('finish_reason')}   (length = cut off by token limit)")
print(f"token usage:    {getattr(resp, 'usage_metadata', None)}")
print(f"extra fields:   {list((resp.additional_kwargs or {}).keys())}   (reasoning_content = thinking sent separately)")
print(f"raw length:     {len(raw)} chars | after cleaning: {len(cleaned)} chars")
print(f"\n--- cleaned reply (first 600 chars) ---\n{cleaned[:600] or '(EMPTY)'}\n")

if not cleaned:
    print("-> EMPTY: the model's answer is missing. Most likely it is a thinking model that used up its")
    print("   tokens thinking. Turn thinking OFF for this model in LM Studio, or raise the context length.")
elif "<think>" in raw:
    print("-> The model is thinking out loud. That works (it is stripped) but is slow; consider turning thinking off.")
elif elapsed > 90:
    print("-> Works, but slow. Tickets may hit the timeout; raise LMSTUDIO_TIMEOUT or use a smaller/faster setup.")
else:
    print("-> LM Studio looks healthy. If drafts are still missing, check the terminal for [LLM] lines.")
