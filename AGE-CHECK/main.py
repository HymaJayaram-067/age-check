from datetime import date, datetime, time

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import os
import json
import re


app = FastAPI(title="Age Check API", version="1.0.0")


class AgeCheckRequest(BaseModel):
    name: str = Field(..., min_length=1, examples=["Ravi"])
    date_of_birth: date = Field(..., examples=["2000-05-10"])
    height_cm: float | None = Field(
        default=None, ge=50, le=250, examples=[175.0], description="Height in centimeters"
    )
    weight_kg: float | None = Field(
        default=None, ge=10, le=250, examples=[72.5], description="Weight in kilograms"
    )
    lifetime_years: int = Field(
        default=100, ge=1, le=200, examples=[100], description="Assumed lifetime years for progress"
    )
    plan_days: int = Field(
        default=14,
        ge=1,
        le=60,
        examples=[14],
        description="Number of days to generate follow-it plan",
    )


class AgeCheckResponse(BaseModel):
    name: str
    years: int
    months: int
    days: int
    hours: int
    minutes: int
    seconds: int
    elapsed_at_epoch_ms: int
    lifetime_years: int
    lifetime_completed_percent: int
    bmi: float | None = None
    bmi_category: str | None = None
    bmi_advice: str | None = None
    diet_advice: str | None = None
    height_cm: float | None = None
    weight_kg: float | None = None
    plan_days: int
    plan_summary: str | None = None
    estimated_change_text: str | None = None
    plan_schedule: list[str] = []
    workout_schedule: list[str] = []


def _calculate_ymd(dob: date, today: date) -> tuple[int, int, int]:
    years = today.year - dob.year
    months = today.month - dob.month
    days = today.day - dob.day

    if days < 0:
        months -= 1
        prev_month = today.month - 1 or 12
        prev_month_year = today.year if today.month > 1 else today.year - 1

        if prev_month == 2:
            leap = (
                prev_month_year % 4 == 0
                and (prev_month_year % 100 != 0 or prev_month_year % 400 == 0)
            )
            days_in_prev_month = 29 if leap else 28
        elif prev_month in {4, 6, 9, 11}:
            days_in_prev_month = 30
        else:
            days_in_prev_month = 31

        days += days_in_prev_month

    if months < 0:
        years -= 1
        months += 12

    return years, months, days


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def _extract_first_json_object(text: str) -> dict | None:
    # Try to locate a JSON object inside the model output.
    try:
        text_stripped = text.strip()
        if text_stripped.startswith("{") and text_stripped.endswith("}"):
            return json.loads(text_stripped)
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _generate_plan_with_ollama(
    *,
    action: str,
    bmi_category: str,
    bmi: float | None,
    height_cm: float | None,
    weight_kg: float | None,
    plan_days: int,
) -> dict | None:
    enabled = _env_bool("ENABLE_LLM", default=False)
    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    if not enabled or provider != "ollama":
        return None

    # Optional dependency; lets local app run even without `requests` installed.
    try:
        import requests  # type: ignore
    except ModuleNotFoundError:
        return None

    base_url = os.getenv("OLLAMA_BASE_URL", "http://age-check-ollama:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "llama3").strip()
    timeout_seconds = int(os.getenv("LLM_TIMEOUT_SECONDS", "60"))

    url = f"{base_url}/api/chat"
    system_prompt = (
        "You are a fitness and nutrition coach. Give a clear follow-it plan based on BMI category. "
        "You MUST output only valid JSON (no markdown, no extra text)."
    )

    prompt = {
        "action": action,
        "bmi_category": bmi_category,
        "bmi": bmi,
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "plan_days": plan_days,
        "output_schema": {
            "plan_summary": "string",
            "estimated_change_text": "string",
            "plan_schedule": ["string"],  # length = plan_days
            "workout_schedule": ["string"],  # length = plan_days
        },
    }

    user_prompt = (
        "Generate a follow-it plan for the next given number of days.\n\n"
        f"Inputs (JSON): {json.dumps(prompt)}\n\n"
        "Rules:\n"
        "- plan_schedule[i] must be diet guidance for Day i+1.\n"
        "- workout_schedule[i] must be workout for Day i+1.\n"
        "- Each day must include something practical (walk/strength/cardio + a diet focus).\n"
        "- Use the action (increase/reduce/maintain) to guide diet.\n"
        f"- Length: plan_schedule and workout_schedule must each have exactly {plan_days} items.\n"
        "- Keep each line short (1-2 sentences).\n"
        "Return ONLY JSON with the exact keys in output_schema."
    )

    try:
        resp = requests.post(
            url,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
            },
            timeout=timeout_seconds,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data.get("message", {}).get("content", "")
        if not content:
            return None
        parsed = _extract_first_json_object(content)
        if not parsed:
            return None
        if "plan_schedule" not in parsed or "workout_schedule" not in parsed:
            return None
        # Ensure lengths match requested plan_days.
        plan_schedule = parsed.get("plan_schedule", [])
        workout_schedule = parsed.get("workout_schedule", [])
        if not isinstance(plan_schedule, list) or not isinstance(workout_schedule, list):
            return None
        if len(plan_schedule) != plan_days or len(workout_schedule) != plan_days:
            return None
        return parsed
    except Exception:
        return None


@app.get("/")
def root() -> HTMLResponse:
    landing = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Age Check</title>
    <style>
      body{
        background: radial-gradient(1200px 600px at 10% 10%, #7c3aed22, transparent),
                    radial-gradient(900px 500px at 90% 20%, #0ea5e922, transparent),
                    linear-gradient(180deg, #0b1220, #070a12);
        color:#e5e7eb;
        font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
        margin:0;
        padding:0;
      }
      .wrap{
        max-width:900px;
        margin:0 auto;
        padding:60px 18px;
      }
      .glass{
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 18px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.35);
        backdrop-filter: blur(10px);
        padding:28px;
      }
      .brand{
        background: linear-gradient(90deg,#7c3aed,#22c55e);
        -webkit-background-clip:text;
        background-clip:text;
        color:transparent;
        font-weight:900;
      }
      .btn{
        display:inline-block;
        padding:12px 18px;
        border:none;
        border-radius:12px;
        cursor:pointer;
        font-weight:800;
        background: linear-gradient(90deg,#7c3aed,#22c55e);
        color:#071018;
        text-decoration:none;
      }
      .muted{ color: rgba(229,231,235,0.75); }
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="glass">
        <div class="muted">FastAPI • Age Calculator</div>
        <h1 style="margin:6px 0 10px 0" class="brand">Hey buddy, welcome to Age Calculation</h1>
        <p class="muted" style="margin:0 0 18px 0">
          Use the dedicated page to enter your details (name, DOB, height, weight) and get your live age + follow-it plan.
        </p>
        <a class="btn" href="/enter">Enter Details</a>
        <div class="muted" style="margin-top:14px; font-size:12px">
          Health: <a class="muted" href="/health" style="color: rgba(229,231,235,0.75)">/health</a>
        </div>
      </div>
    </div>
  </body>
</html>
    """
    return HTMLResponse(content=landing)


@app.get("/enter", response_class=HTMLResponse)
def enter_ui() -> str:
    # Dedicated page for users to enter details.
    # After submit we redirect to `/result` with the API response payload.
    return """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Enter Details - Age Check</title>
    <style>
      body{
        background: radial-gradient(1200px 600px at 10% 10%, #7c3aed22, transparent),
                    radial-gradient(900px 500px at 90% 20%, #0ea5e922, transparent),
                    linear-gradient(180deg, #0b1220, #070a12);
        color:#e5e7eb;
        font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
        margin:0;
        padding:0;
      }
      .wrap{ max-width: 960px; margin:0 auto; padding:56px 18px; }
      .glass{
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 18px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.35);
        backdrop-filter: blur(10px);
        padding: 28px;
      }
      .brand{
        background: linear-gradient(90deg,#7c3aed,#22c55e);
        -webkit-background-clip:text;
        background-clip:text;
        color:transparent;
        font-weight:900;
      }
      .muted{ color: rgba(229,231,235,0.75); }
      .grid{ display:grid; grid-template-columns: 1fr; gap: 14px; }
      @media (min-width: 900px){
        .grid-2{ grid-template-columns: 1fr 1fr; }
      }
      label{ display:block; font-weight:700; margin-bottom:6px; color: rgba(229,231,235,0.95); }
      input, button{
        width:100%;
        box-sizing:border-box;
        padding: 12px 14px;
        border-radius: 12px;
      }
      input{
        border: 1px solid rgba(255,255,255,0.18);
        background: rgba(255,255,255,0.04);
        color: #e5e7eb;
        outline: none;
      }
      input:focus{
        border-color: rgba(34,197,94,0.55);
        box-shadow: 0 0 0 3px rgba(124,58,237,0.18);
      }
      .btn{
        border:none;
        cursor:pointer;
        font-weight:900;
        background: linear-gradient(90deg,#7c3aed,#22c55e);
        color:#071018;
        margin-top: 6px;
      }
      .error{ color:#f87171; font-weight:700; margin-top: 10px; }
      .cardNote{
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 12px 14px;
      }
      .small{ font-size: 12px; }
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="glass">
        <div style="display:flex; justify-content:space-between; gap:16px; align-items:flex-start; flex-wrap:wrap;">
          <div>
            <div class="muted">FastAPI • Age Calculator</div>
            <h1 class="brand" style="margin:6px 0 10px 0;">Enter Your Details</h1>
            <div class="muted">After you click Calculate, you will be sent to a result page with all outputs.</div>
          </div>
          <div class="small muted" style="text-align:right; min-width: 160px;">
            Local Server<br/>`/health` available
          </div>
        </div>

        <div style="height:18px;"></div>

        <form id="ageForm">
          <div class="grid grid-2">
            <div>
              <label for="name">Name</label>
              <input id="name" name="name" type="text" placeholder="Ravi" required />
            </div>

            <div>
              <label for="dob">Date of Birth</label>
              <input id="dob" name="date_of_birth" type="date" required />
            </div>
          </div>

          <div style="height:14px;"></div>

          <div class="grid grid-2">
            <div>
              <label for="height_cm">Height (cm)</label>
              <input id="height_cm" name="height_cm" type="number" placeholder="175" min="50" max="250" step="0.1" />
              <div class="small muted" style="margin-top:6px;">Optional for BMI + plan</div>
            </div>

            <div>
              <label for="weight_kg">Weight (kg)</label>
              <input id="weight_kg" name="weight_kg" type="number" placeholder="72.5" min="10" max="250" step="0.1" />
              <div class="small muted" style="margin-top:6px;">Optional for BMI + plan</div>
            </div>
          </div>

          <div style="height:14px;"></div>

          <div class="grid grid-2">
            <div>
              <label for="lifetime_years">Lifetime (Years) - water fill</label>
              <input id="lifetime_years" name="lifetime_years" type="number" value="100" min="1" max="200" step="1" />
            </div>

            <div>
              <label for="plan_days">Follow-it Plan Days</label>
              <input id="plan_days" name="plan_days" type="number" value="14" min="1" max="60" step="1" />
            </div>
          </div>

          <div style="height:18px;"></div>

          <button id="calcBtn" class="btn" type="submit">Calculate</button>
          <div id="error" class="error" style="display:none;"></div>

          <div style="height:10px;"></div>
          <div class="cardNote">
            <div class="small muted">
              Tip: If you want AI-generated plans in Minikube, enable LLM via environment (Ollama). Locally it will use rule-based plan.
            </div>
          </div>
        </form>
      </div>
    </div>

    <script>
      function base64EncodeUnicode(str) {
        return btoa(unescape(encodeURIComponent(str)));
      }

      function showError(msg) {
        const errorEl = document.getElementById('error');
        errorEl.textContent = msg;
        errorEl.style.display = 'block';
      }

      document.getElementById('ageForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        showError('');
        document.getElementById('error').style.display = 'none';

        const name = document.getElementById('name').value.trim();
        const dob = document.getElementById('dob').value;
        const heightRaw = document.getElementById('height_cm').value;
        const weightRaw = document.getElementById('weight_kg').value;
        const lifetimeYearsRaw = document.getElementById('lifetime_years').value;
        const planDaysRaw = document.getElementById('plan_days').value;

        const payload = {
          name: name,
          date_of_birth: dob,
          lifetime_years: lifetimeYearsRaw ? parseInt(lifetimeYearsRaw) : 100,
          plan_days: planDaysRaw ? parseInt(planDaysRaw) : 14
        };
        if (heightRaw) payload.height_cm = parseFloat(heightRaw);
        if (weightRaw) payload.weight_kg = parseFloat(weightRaw);

        try {
          const resp = await fetch('/age/check', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
          });
          const data = await resp.json();
          if (!resp.ok) {
            showError(data && data.detail ? data.detail : 'Request failed');
            return;
          }

          // Redirect to results page with encoded JSON
          const json = JSON.stringify(data);
          const encoded = base64EncodeUnicode(json);
          window.location.href = '/result?payload=' + encodeURIComponent(encoded);
        } catch (err) {
          showError('Network error. Make sure the server is running.');
        }
      });
    </script>
  </body>
</html>
    """


@app.get("/result", response_class=HTMLResponse)
def result_ui() -> str:
    # Result page reads `payload` from query string.
    return """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Age Result</title>
    <style>
      body{
        background: radial-gradient(1200px 600px at 10% 10%, #7c3aed22, transparent),
                    radial-gradient(900px 500px at 90% 20%, #0ea5e922, transparent),
                    linear-gradient(180deg, #0b1220, #070a12);
        color:#e5e7eb;
        font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
        margin:0;
        padding:0;
      }
      .wrap{ max-width: 1100px; margin:0 auto; padding:40px 18px; }
      .glass{
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 18px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.35);
        backdrop-filter: blur(10px);
        padding: 28px;
      }
      .brand{
        background: linear-gradient(90deg,#7c3aed,#22c55e);
        -webkit-background-clip:text;
        background-clip:text;
        color:transparent;
        font-weight:900;
      }
      .muted{ color: rgba(229,231,235,0.75); }
      .btn{
        border:none;
        cursor:pointer;
        font-weight:900;
        background: linear-gradient(90deg,#7c3aed,#22c55e);
        color:#071018;
        padding: 12px 18px;
        border-radius: 12px;
        text-decoration:none;
        display:inline-block;
      }
      .error{ color:#f87171; font-weight:700; margin-top: 10px; }
      .metric{
        border-radius: 14px;
        padding: 12px 14px;
        border: 1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.04);
      }
      .metric .label{ color: rgba(203,213,225,0.95); font-size: 12px; font-weight: 700; }
      .metric .value{ font-size: 26px; font-weight: 900; margin-top: 4px; }
      .badge-soft{
        border-radius: 999px;
        padding: 8px 12px;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        display:inline-block;
      }
      .waterWrap{
        position: relative;
        height: 220px;
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.14);
        background: rgba(255,255,255,0.03);
        overflow: hidden;
      }
      .waterFill{
        position: absolute;
        left: 0;
        bottom: 0;
        width: 100%;
        height: 0%;
        background: linear-gradient(180deg, rgba(34,197,94,0.65), rgba(124,58,237,0.55));
        transition: height 0.6s ease;
      }
      .waterFill::before{
        content: "";
        position: absolute;
        top: -30px;
        left: 0;
        width: 200%;
        height: 70px;
        background: radial-gradient(circle at 20% 30%, rgba(255,255,255,0.45), transparent 40%),
                    linear-gradient(90deg, rgba(255,255,255,0.25), rgba(255,255,255,0.05));
        opacity: 0.55;
        animation: wave 2.6s linear infinite;
      }
      @keyframes wave{
        0% { transform: translateX(-50px); }
        100% { transform: translateX(50px); }
      }
      .waterText{
        position: relative;
        z-index: 2;
        padding: 18px;
        display:flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100%;
      }
      .waterText .big{ font-size: 34px; font-weight: 900; letter-spacing: -0.02em; }
      .waterText .small{ font-size: 12px; color: rgba(229,231,235,0.85); }
      .grid2{ display:grid; grid-template-columns: 1fr; gap:14px; }
      @media (min-width: 900px){
        .grid2{ grid-template-columns: 0.45fr 0.55fr; }
      }
      .planCard{
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 14px;
        margin-bottom: 10px;
      }
      .text-primary{ color: rgba(96,165,250,0.95); }
    </style>
  </head>
  <body>
    <div class="wrap">
      <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:16px; flex-wrap:wrap;">
        <div>
          <div class="badge-soft mb-2">FastAPI • Age Calculator</div>
          <h1 class="brand" style="margin:6px 0 8px 0;">Result Dashboard</h1>
          <div class="muted">Live seconds clock runs on this result page.</div>
        </div>
        <div>
          <a class="btn" href="/enter">Back to Enter</a>
        </div>
      </div>

      <div style="height:18px;"></div>
      <div id="error" class="error d-none" style="display:none;"></div>

      <div id="content" class="grid2">
        <div class="waterWrap">
          <div class="waterFill" id="waterFill"></div>
          <div class="waterText">
            <div>
              <div class="badge-soft mb-2 w-100" style="width:fit-content;">Lifetime Water Fill</div>
              <div class="small" id="lifetimeNote"></div>
            </div>
            <div>
              <div class="small">Completed</div>
              <div class="big" id="lifetimePercent">0%</div>
              <div class="small" id="lifetimeYearsText"></div>
            </div>
          </div>
        </div>

        <div>
          <div style="display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:12px;">
            <div class="metric"><div class="label">Years</div><div class="value" id="yearsVal">0</div></div>
            <div class="metric"><div class="label">Months</div><div class="value" id="monthsVal">0</div></div>
            <div class="metric"><div class="label">Days</div><div class="value" id="daysVal">0</div></div>
            <div class="metric"><div class="label">Hours (live)</div><div class="value" id="hoursVal">0</div></div>
            <div class="metric"><div class="label">Minutes (live)</div><div class="value" id="minutesVal">0</div></div>
            <div class="metric"><div class="label">Seconds (live)</div><div class="value" id="secondsVal">0</div></div>
          </div>

          <div style="height:12px;"></div>
          <div class="planCard">
            <div style="font-weight:900;">Age Summary</div>
            <div class="muted" id="ageSummary">-</div>
          </div>

          <div style="height:12px;"></div>
          <div class="planCard">
            <div style="font-weight:900;">BMI + Health Check</div>
            <div class="badge-soft" id="bmiCategory">BMI: -</div>
            <div class="muted" id="bmiAdvice" style="margin-top:8px;"></div>
          </div>

          <div style="height:12px;"></div>
          <div class="planCard">
            <div style="font-weight:900;">Diet + Workout Follow-it Plan</div>
            <div class="muted" id="dietAdvice" style="white-space:pre-wrap;margin-top:6px;"></div>
            <div id="estimatedChangeText" class="muted" style="margin-top:10px;"></div>
            <div class="muted" style="margin-top:12px;">Plan schedule:</div>
            <div id="planCards" style="margin-top:10px;"></div>
          </div>
        </div>
      </div>
    </div>

    <script>
      function base64DecodeUnicode(b64) {
        return decodeURIComponent(escape(atob(b64)));
      }

      const params = new URLSearchParams(window.location.search);
      const payloadParam = params.get('payload');
      const errorEl = document.getElementById('error');
      const contentEl = document.getElementById('content');

      if (!payloadParam) {
        errorEl.textContent = 'Missing payload. Please go back to /enter.';
        errorEl.style.display = 'block';
        contentEl.style.display = 'none';
      } else {
        let data = null;
        try {
          data = JSON.parse(base64DecodeUnicode(payloadParam));
        } catch (e) {
          errorEl.textContent = 'Invalid payload. Please go back to /enter.';
          errorEl.style.display = 'block';
          contentEl.style.display = 'none';
        }

        if (data) {
          // Age summary
          document.getElementById('yearsVal').textContent = data.years;
          document.getElementById('monthsVal').textContent = data.months;
          document.getElementById('daysVal').textContent = data.days;

          const n = data.name || '';
          document.getElementById('ageSummary').textContent =
            n + ' is ' + data.years + ' years, ' + data.months + ' months, and ' + data.days + ' days old.';

          // Lifetime water fill
          const percent = data.lifetime_completed_percent ?? 0;
          document.getElementById('lifetimePercent').textContent = percent + '%';
          document.getElementById('lifetimeYearsText').textContent = 'Approx: ' + percent + '% of ' + data.lifetime_years + ' years';
          document.getElementById('lifetimeNote').textContent = 'Assumed lifetime: ' + data.lifetime_years + ' years';
          document.getElementById('waterFill').style.height = percent + '%';

          // BMI
          if (data.bmi_category) {
            const bmiText = data.bmi !== null && data.bmi !== undefined ? data.bmi.toFixed(1) : '';
            document.getElementById('bmiCategory').textContent =
              'BMI: ' + data.bmi_category + (bmiText ? (' (' + bmiText + ')') : '');
          }
          if (data.height_cm && data.weight_kg) {
            document.getElementById('bmiAdvice').textContent =
              'For ' + data.height_cm + ' cm & ' + data.weight_kg + ' kg: ' + (data.bmi_advice || '');
          } else {
            document.getElementById('bmiAdvice').textContent =
              data.bmi_advice ? data.bmi_advice : 'Enter height and weight to see health guidance.';
          }

          // Diet follow-it plan
          document.getElementById('dietAdvice').textContent =
            data.plan_summary ? data.plan_summary : (data.diet_advice || '');
          document.getElementById('estimatedChangeText').textContent =
            data.estimated_change_text ? data.estimated_change_text : 'Estimated change: -';

          const planCards = document.getElementById('planCards');
          planCards.innerHTML = '';

          const schedule = data.plan_schedule || [];
          const workouts = data.workout_schedule || [];
          const days = data.plan_days || schedule.length;
          const showDays = Math.min(days, schedule.length, workouts.length, 30);

          if (showDays > 0) {
            for (let i = 0; i < showDays; i++) {
              const dayNum = i + 1;
              const dietLine = schedule[i] || '';
              const workoutLine = workouts[i] || '';

              const card = document.createElement('div');
              card.className = 'planCard';
              card.style.padding = '12px 14px';
              card.innerHTML = `
                <div style="font-weight:900; margin-bottom:6px;">Day ${dayNum}</div>
                <div class="muted" style="white-space:pre-wrap;">${dietLine}</div>
                <div class="text-primary" style="margin-top:8px; font-weight:800;">
                  Workout: ${workoutLine}
                </div>
              `;
              planCards.appendChild(card);
            }

            if (days > showDays) {
              const more = document.createElement('div');
              more.className = 'muted';
              more.style.marginTop = '8px';
              more.textContent = 'Showing first ' + showDays + ' days. Total plan days: ' + days + '.';
              planCards.appendChild(more);
            }
          } else {
            planCards.innerHTML = '<div class="muted">Enter height + weight to generate a follow-it plan.</div>';
          }

          // Live seconds clock
          let baseElapsedSeconds = data.seconds || 0;
          let elapsedAtMs = data.elapsed_at_epoch_ms || Date.now();

          const secondsVal = document.getElementById('secondsVal');
          const minutesVal = document.getElementById('minutesVal');
          const hoursVal = document.getElementById('hoursVal');

          const tick = () => {
            const deltaSeconds = Math.floor((Date.now() - elapsedAtMs) / 1000);
            const elapsed = baseElapsedSeconds + deltaSeconds;
            secondsVal.textContent = elapsed;
            minutesVal.textContent = Math.floor(elapsed / 60);
            hoursVal.textContent = Math.floor(elapsed / 3600);
          };

          tick();
          setInterval(tick, 1000);
        }
      }
    </script>
  </body>
</html>
    """


@app.get("/health")
def health() -> dict[str, str]:
    return {"message": "Age Check API is running"}


@app.get("/ui", response_class=HTMLResponse)
def age_ui() -> str:
    return """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Age Check UI</title>
    <!-- External CDN removed to prevent browser "loading forever" when internet is blocked. -->
    <style>
      body {
        background: radial-gradient(1200px 600px at 10% 10%, #7c3aed22, transparent),
                    radial-gradient(900px 500px at 90% 20%, #0ea5e922, transparent),
                    linear-gradient(180deg, #0b1220, #070a12);
        color: #e5e7eb;
        font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
      }
      .glass {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 18px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.35);
        backdrop-filter: blur(10px);
      }
      .brand {
        background: linear-gradient(90deg, #7c3aed, #22c55e);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        font-weight: 800;
      }
      .badge-soft {
        border-radius: 999px;
        padding: 8px 12px;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
      }
      .metric {
        border-radius: 14px;
        padding: 12px 14px;
        border: 1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.04);
      }
      .metric .label { color: #cbd5e1; font-size: 12px; font-weight: 600; }
      .metric .value { font-size: 26px; font-weight: 800; margin-top: 4px; }
      .error { color: #f87171; }

      /* Water fill lifetime progress */
      .waterWrap {
        position: relative;
        height: 220px;
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.14);
        background: rgba(255,255,255,0.03);
        overflow: hidden;
      }
      .waterFill {
        position: absolute;
        left: 0;
        bottom: 0;
        width: 100%;
        height: 40%;
        background: linear-gradient(180deg, rgba(34,197,94,0.65), rgba(124,58,237,0.55));
        transition: height 0.6s ease;
      }
      .waterFill::before {
        content: "";
        position: absolute;
        top: -30px;
        left: 0;
        width: 200%;
        height: 70px;
        background: radial-gradient(circle at 20% 30%, rgba(255,255,255,0.45), transparent 40%),
                    linear-gradient(90deg, rgba(255,255,255,0.25), rgba(255,255,255,0.05));
        opacity: 0.55;
        animation: wave 2.6s linear infinite;
      }
      @keyframes wave {
        0% { transform: translateX(-50px); }
        100% { transform: translateX(50px); }
      }
      .waterText {
        position: relative;
        z-index: 2;
        padding: 18px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100%;
      }
      .waterText .small { color: rgba(229,231,235,0.85); font-size: 12px; }
      .waterText .big { font-size: 34px; font-weight: 900; letter-spacing: -0.02em; }

      .adviceBox {
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.04);
        padding: 14px;
      }

      /* Bootstrap fallback (in case CDN is blocked) */
      .text-secondary { color: rgba(229,231,235,0.75) !important; }
      .text-success { color: #22c55e !important; }
      .text-info { color: #38bdf8 !important; }
      .text-warning { color: #fbbf24 !important; }
      .text-primary { color: #60a5fa !important; }
      .text-danger { color: #f87171 !important; }
      .fw-semibold { font-weight: 600 !important; }
      .fw-bold { font-weight: 800 !important; }
      .small { font-size: 12px; color: rgba(229,231,235,0.85); }
      .fs-5 { font-size: 1.25rem; }
      .fs-6 { font-size: 1.02rem; }
      .form-text { font-size: 12px; color: rgba(229,231,235,0.7); }
      .d-none { display: none !important; }
      .badge-soft { line-height: 1.1; }
    </style>
  </head>
  <body>
    <div class="container py-5">
      <div class="row justify-content-center">
        <div class="col-12 col-lg-10">
          <div class="glass p-4 p-md-5">
            <div class="d-flex align-items-start justify-content-between gap-3">
              <div>
                <div class="badge-soft mb-2">FastAPI • Age Calculator</div>
                <h1 class="h3 mb-1 brand">Hey buddy, welcome to Age Calculation</h1>
                <div class="text-secondary mb-4">Enter name + DOB (and optionally height/weight) to get a live seconds clock and lifetime progress.</div>
              </div>
              <div class="text-end">
                <div class="small text-secondary">Server</div>
                <div class="fw-semibold">Localhost</div>
              </div>
            </div>

            <form id="ageForm" class="row g-3">
              <div class="col-12 col-md-4">
                <label class="form-label fw-semibold" for="name">Name</label>
                <input id="name" name="name" type="text" class="form-control form-control-lg" placeholder="Ravi" required>
              </div>

              <div class="col-12 col-md-4">
                <label class="form-label fw-semibold" for="dob">Date of Birth</label>
                <input id="dob" name="date_of_birth" type="date" class="form-control form-control-lg" required>
                <div class="form-text text-secondary">Future dates are rejected.</div>
              </div>

              <div class="col-12 col-md-3">
                <label class="form-label fw-semibold" for="lifetime_years">Lifetime (Years)</label>
                <input id="lifetime_years" name="lifetime_years" type="number" class="form-control form-control-lg" value="100" min="1" max="200" step="1">
                <div class="form-text text-secondary">Example: 100</div>

                <label class="form-label fw-semibold mt-3" for="plan_days">Follow Plan (Days)</label>
                <input id="plan_days" name="plan_days" type="number" class="form-control form-control-lg" value="14" min="1" max="60" step="1">
                <div class="form-text text-secondary">Example: 14</div>
              </div>

              <div class="col-12 col-md-3">
                <label class="form-label fw-semibold" for="height_cm">Height (cm)</label>
                <input id="height_cm" name="height_cm" type="number" class="form-control form-control-lg" placeholder="175" min="50" max="250" step="0.1">
              </div>

              <div class="col-12 col-md-3">
                <label class="form-label fw-semibold" for="weight_kg">Weight (kg)</label>
                <input id="weight_kg" name="weight_kg" type="number" class="form-control form-control-lg" placeholder="72.5" min="10" max="250" step="0.1">
              </div>

              <div class="col-12 col-md-3 d-grid">
                <button type="submit" id="calcBtn" class="btn btn-lg btn-primary mt-4"
                        style="background: linear-gradient(90deg,#7c3aed,#22c55e); border:none;">
                  <span id="calcText">Calculate</span>
                  <span id="spinner" class="spinner-border spinner-border-sm ms-2 d-none" role="status" aria-hidden="true"></span>
                </button>
              </div>
            </form>

            <div id="error" class="error mt-3 d-none"></div>

            <div id="resultWrap" class="mt-4 d-none">
              <div class="row g-3">
                <div class="col-12 col-lg-4">
                  <div class="waterWrap">
                    <div class="waterFill" id="waterFill"></div>
                    <div class="waterText">
                      <div>
                        <div class="badge-soft mb-3 w-100 text-center">Lifetime Water Fill</div>
                        <div class="small" id="lifetimeNote">Assumed lifetime progress</div>
                      </div>
                      <div>
                        <div class="small">Completed</div>
                        <div class="big" id="lifetimePercent">0%</div>
                        <div class="small" id="lifetimeYearsText">0 / 100 years</div>
                      </div>
                    </div>
                  </div>
                </div>

                <div class="col-12 col-lg-8">
                  <div class="p-3 rounded-3" style="background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12);">
                    <div class="fs-5 fw-bold mb-2" id="ageSummary">Age Summary</div>
                    <div class="row g-3">
                      <div class="col-12 col-md-4">
                        <div class="metric">
                          <div class="label">Years</div>
                          <div class="value text-success" id="yearsVal">0</div>
                        </div>
                      </div>
                      <div class="col-12 col-md-4">
                        <div class="metric">
                          <div class="label">Months</div>
                          <div class="value text-info" id="monthsVal">0</div>
                        </div>
                      </div>
                      <div class="col-12 col-md-4">
                        <div class="metric">
                          <div class="label">Days</div>
                          <div class="value text-warning" id="daysVal">0</div>
                        </div>
                      </div>
                      <div class="col-12 col-md-4">
                        <div class="metric">
                          <div class="label">Hours (live)</div>
                          <div class="value text-primary" id="hoursVal">0</div>
                        </div>
                      </div>
                      <div class="col-12 col-md-4">
                        <div class="metric">
                          <div class="label">Minutes (live)</div>
                          <div class="value text-secondary" id="minutesVal">0</div>
                        </div>
                      </div>
                      <div class="col-12 col-md-4">
                        <div class="metric">
                          <div class="label">Seconds (live)</div>
                          <div class="value text-danger" id="secondsVal">0</div>
                        </div>
                      </div>
                    </div>

                    <div class="text-secondary small mt-3">
                      Live clock updates from the server time when you press <b>Calculate</b>.
                    </div>
                  </div>

                  <div class="row g-3 mt-3">
                    <div class="col-12 col-md-6">
                      <div class="adviceBox">
                        <div class="fs-6 fw-bold mb-2">Height + Weight Health Check</div>
                        <div class="small text-secondary mb-2">Enter both height and weight to get BMI advice.</div>
                        <div class="badge-soft mb-2" id="bmiCategory">BMI: -</div>
                        <div id="bmiAdvice" class="text-secondary"></div>
                      </div>
                    </div>
                    <div class="col-12 col-md-6">
                      <div class="adviceBox">
                        <div class="fs-6 fw-bold mb-2">Diet Advice</div>
                        <div class="small text-secondary mb-2">
                          This is rule-based guidance (not a medical diagnosis). For AI diet prediction, we can integrate an LLM later.
                        </div>
                        <div id="dietAdvice" class="text-secondary"></div>
                        <div class="badge-soft mt-3 mb-3" id="estimatedChangeText">Estimated change: -</div>
                        <div class="small text-secondary mb-2">Your follow-it plan (diet + workouts):</div>
                        <div id="planCards" class="mt-2"></div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <script>
      const form = document.getElementById('ageForm');
      const errorEl = document.getElementById('error');
      const resultWrap = document.getElementById('resultWrap');

      const calcBtn = document.getElementById('calcBtn');
      const calcText = document.getElementById('calcText');
      const spinner = document.getElementById('spinner');

      const ageSummary = document.getElementById('ageSummary');
      const yearsVal = document.getElementById('yearsVal');
      const monthsVal = document.getElementById('monthsVal');
      const daysVal = document.getElementById('daysVal');
      const hoursVal = document.getElementById('hoursVal');
      const minutesVal = document.getElementById('minutesVal');
      const secondsVal = document.getElementById('secondsVal');

      const waterFill = document.getElementById('waterFill');
      const lifetimePercent = document.getElementById('lifetimePercent');
      const lifetimeYearsText = document.getElementById('lifetimeYearsText');
      const lifetimeNote = document.getElementById('lifetimeNote');

      const bmiCategory = document.getElementById('bmiCategory');
      const bmiAdvice = document.getElementById('bmiAdvice');
      const dietAdvice = document.getElementById('dietAdvice');
      const estimatedChangeText = document.getElementById('estimatedChangeText');
      const planCards = document.getElementById('planCards');

      let liveTimer = null;

      let baseElapsedSeconds = 0;
      let elapsedAtMs = 0;

      function showError(msg) {
        errorEl.textContent = msg;
        errorEl.classList.remove('d-none');
      }

      function setLoading(isLoading) {
        calcBtn.disabled = isLoading;
        if (isLoading) {
          calcText.textContent = 'Calculating';
          spinner.classList.remove('d-none');
        } else {
          calcText.textContent = 'Calculate';
          spinner.classList.add('d-none');
        }
      }

      function setLiveClock(data) {
        baseElapsedSeconds = data.seconds;
        elapsedAtMs = data.elapsed_at_epoch_ms;

        if (liveTimer) clearInterval(liveTimer);

        const tick = () => {
          const deltaSeconds = Math.floor((Date.now() - elapsedAtMs) / 1000);
          const elapsed = baseElapsedSeconds + deltaSeconds;
          secondsVal.textContent = elapsed;
          minutesVal.textContent = Math.floor(elapsed / 60);
          hoursVal.textContent = Math.floor(elapsed / 3600);
        };

        tick();
        liveTimer = setInterval(tick, 1000);
      }

      form.addEventListener('submit', async (e) => {
        e.preventDefault();
        resultWrap.classList.add('d-none');
        errorEl.classList.add('d-none');
        setLoading(true);

        const name = document.getElementById('name').value.trim();
        const dob = document.getElementById('dob').value;

        const heightRaw = document.getElementById('height_cm').value;
        const weightRaw = document.getElementById('weight_kg').value;
        const lifetimeYearsRaw = document.getElementById('lifetime_years').value;
        const planDaysRaw = document.getElementById('plan_days').value;

        const payload = {
          name: name,
          date_of_birth: dob,
          lifetime_years: lifetimeYearsRaw ? parseInt(lifetimeYearsRaw) : 100,
          plan_days: planDaysRaw ? parseInt(planDaysRaw) : 14,
        };

        if (heightRaw) payload.height_cm = parseFloat(heightRaw);
        if (weightRaw) payload.weight_kg = parseFloat(weightRaw);

        try {
          const resp = await fetch('/age/check', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
          });

          const data = await resp.json();

          if (!resp.ok) {
            showError(data && data.detail ? data.detail : 'Request failed');
            setLoading(false);
            return;
          }

          // Age part (years/months/days are fixed at calculation time)
          yearsVal.textContent = data.years;
          monthsVal.textContent = data.months;
          daysVal.textContent = data.days;

          const n = data.name || name;
          ageSummary.textContent = n + ' is ' + data.years + ' years, ' + data.months + ' months, and ' + data.days + ' days old.';

          // Lifetime progress water fill
          const percent = data.lifetime_completed_percent;
          lifetimePercent.textContent = percent + '%';
          lifetimeYearsText.textContent = 'Approx: ' + percent + '% of ' + data.lifetime_years + ' years';
          waterFill.style.height = percent + '%';
          lifetimeNote.textContent = 'Assumed lifetime: ' + data.lifetime_years + ' years';

          // BMI + diet advice
          if (data.bmi_category) {
            const bmiText = data.bmi !== null && data.bmi !== undefined ? data.bmi.toFixed(1) : '';
            bmiCategory.textContent = 'BMI: ' + data.bmi_category + (bmiText ? (' (' + bmiText + ')') : '');
          } else {
            bmiCategory.textContent = 'BMI: - (enter height & weight)';
          }

          if (data.height_cm && data.weight_kg) {
            bmiAdvice.textContent =
              'For ' + data.height_cm + ' cm & ' + data.weight_kg + ' kg: ' +
              (data.bmi_advice ? data.bmi_advice : '');
          } else {
            bmiAdvice.textContent = data.bmi_advice ? data.bmi_advice : 'Enter height and weight to see personalized advice.';
          }

          // BMI health check + plan
          dietAdvice.textContent = data.plan_summary ? data.plan_summary : (data.diet_advice || 'Enter height and weight.');

          if (data.estimated_change_text) {
            estimatedChangeText.textContent = data.estimated_change_text;
          } else {
            estimatedChangeText.textContent = 'Estimated change: -';
          }

          // Build day cards (diet + workout)
          planCards.innerHTML = '';
          const days = data.plan_days || 0;
          const schedule = data.plan_schedule || [];
          const workouts = data.workout_schedule || [];
          const showDays = Math.min(days, schedule.length, workouts.length, 30); // keep UI usable

          if (showDays > 0) {
            for (let i = 0; i < showDays; i++) {
              const dayNum = i + 1;
              const dietLine = schedule[i] || '';
              const workoutLine = workouts[i] || '';

              const card = document.createElement('div');
              card.className = 'p-3 rounded-3';
              card.style.background = 'rgba(255,255,255,0.04)';
              card.style.border = '1px solid rgba(255,255,255,0.12)';
              card.style.marginBottom = '10px';

              card.innerHTML = `
                <div class="fw-bold mb-2">Day ${dayNum}</div>
                <div class="small text-secondary" style="white-space: pre-wrap;">${dietLine}</div>
                <div class="small mt-2" style="color: rgba(96,165,250,0.95); font-weight: 600;">
                  Workout: ${workoutLine}
                </div>
              `;
              planCards.appendChild(card);
            }

            if (days > showDays) {
              const more = document.createElement('div');
              more.className = 'small text-secondary mt-1';
              more.textContent = `Showing first ${showDays} days. Total plan days: ${days}.`;
              planCards.appendChild(more);
            }
          } else {
            planCards.innerHTML = '<div class="small text-secondary">Enter height and weight to generate your plan.</div>';
          }

          // Start live seconds clock
          setLiveClock(data);

          resultWrap.classList.remove('d-none');
        } catch (err) {
          showError('Network error. Make sure the server is running.');
        } finally {
          setLoading(false);
        }
      });
    </script>
  </body>
</html>
"""


@app.post("/age/check", response_model=AgeCheckResponse)
def check_age(payload: AgeCheckRequest) -> AgeCheckResponse:
    today = date.today()
    if payload.date_of_birth > today:
        raise HTTPException(status_code=400, detail="date_of_birth cannot be in the future")

    years, months, days = _calculate_ymd(payload.date_of_birth, today)

    birth_dt = datetime.combine(payload.date_of_birth, time.min)
    now_dt = datetime.now()
    total_seconds = int((now_dt - birth_dt).total_seconds())
    total_minutes = total_seconds // 60
    total_hours = total_minutes // 60

    # Lifetime progress is an approximation using years + fraction from months/days.
    lifetime_years = payload.lifetime_years
    approx_years = years + (months / 12.0) + (days / 365.25)
    lifetime_completed_percent = int(
        max(0.0, min(100.0, (approx_years / float(lifetime_years)) * 100.0))
    )

    bmi: float | None = None
    bmi_category: str | None = None
    bmi_advice: str | None = None
    diet_advice: str | None = None
    if payload.height_cm and payload.weight_kg:
        height_m = payload.height_cm / 100.0
        if height_m > 0:
            bmi = payload.weight_kg / (height_m * height_m)
            # Simple BMI categories (adult-style). For children/teens BMI percentiles should be used.
            if bmi < 18.5:
                bmi_category = "Underweight"
                bmi_advice = (
                    "Your BMI looks low. Consider eating a calorie-appropriate diet with enough protein, "
                    "healthy carbs, and strength training. If needed, consult a doctor/dietitian."
                )
                diet_advice = (
                    "Diet tip: add nutrient-dense calories (nuts, eggs, dairy/curd), eat 3 meals + 1 snack, "
                    "include protein every meal, and do resistance training 3-4 days/week."
                )
            elif bmi < 25.0:
                bmi_category = "Healthy range"
                bmi_advice = (
                    "Great! Your BMI is in a healthy range. Maintaining gooood, keep it up (steady activity + balanced diet)."
                )
                diet_advice = (
                    "Diet tip: keep doing what you do—focus on whole foods, adequate protein, plenty of vegetables, "
                    "and maintain regular movement."
                )
            elif bmi < 30.0:
                bmi_category = "Overweight"
                bmi_advice = (
                    "Your BMI is above the healthy range. To improve: reduce sugary/processed foods, control portion sizes, "
                    "increase vegetables + lean protein, and add regular walking/cardio."
                )
                diet_advice = (
                    "Diet tip: swap sugary drinks to water, reduce refined carbs, build meals with half vegetables, "
                    "lean protein, and smart carbs; aim for a small calorie deficit."
                )
            else:
                bmi_category = "Obesity"
                bmi_advice = (
                    "Your BMI is high. To improve: follow a calorie-reduction plan, choose whole foods, "
                    "limit sugar/refined carbs, stay active daily, and consider medical guidance."
                )
                diet_advice = (
                    "Diet tip: choose high-fiber foods, limit sweets and fried items, track portions, "
                    "prioritize protein + vegetables, and consider a structured plan with a clinician if needed."
                )

    # Simple "AI coach" plan (offline rule-based) based on BMI category.
    # This is not medical advice; it gives general fitness + diet structure.
    plan_days = payload.plan_days
    plan_summary: str | None = None
    estimated_change_text: str | None = None
    plan_schedule: list[str] = []
    workout_schedule: list[str] = []

    action: str | None = None  # "reduce", "increase", "maintain"
    if bmi_category == "Underweight":
        action = "increase"
    elif bmi_category in {"Overweight", "Obesity"}:
        action = "reduce"
    elif bmi_category == "Healthy range":
        action = "maintain"

    if action is None:
        plan_summary = "Enter height and weight to see an AI-style diet + workout follow-it plan."
    else:
        # Rough estimate: 0.25-0.5 kg per week is typical for body-weight changes.
        # We choose a mid value for user guidance only.
        kg_per_week = 0.35
        weeks = plan_days / 7.0
        est_kg = kg_per_week * weeks
        sign = 1.0 if action == "increase" else (-1.0 if action == "reduce" else 0.0)
        if sign == 0.0:
            estimated_change_text = f"Estimated change: about 0 kg in {plan_days} days (maintain goal)."
        else:
            delta = est_kg * sign
            arrow = "increase" if delta > 0 else "reduce"
            estimated_change_text = (
                f"Estimated {arrow}: ~{abs(delta):.1f} kg over {plan_days} days (approximation)."
            )

        baseDietLine = ""
        if action == "reduce":
            baseDietLine = (
                "Focus on a small calorie deficit: half plate vegetables, lean protein each meal, "
                "limit sugary drinks + refined carbs, and drink water."
            )
        elif action == "increase":
            baseDietLine = (
                "Focus on a small calorie surplus: add extra protein + healthy carbs, eat 3 meals + 1 snack, "
                "and include calorie-dense nutritious foods (nuts, curd/milk, eggs, whole grains)."
            )
        else:
            baseDietLine = (
                "Focus on balanced maintenance: regular protein, vegetables, whole grains, and consistent activity."
            )

        # Workout patterns (repeat every 7 days)
        reducePlan = [
            "Day type A: Brisk walk 30 min + light stretching",
            "Day type B: Strength (full body) 30-40 min (squats/rows/pushups)",
            "Day type C: Interval walk 20 min (easy + hard pace) + core 10 min",
            "Day type B: Strength 30-40 min (focus on legs + back)",
            "Day type A: Walk 35 min + mobility",
            "Day type D: Cardio 25 min + gentle strength 15 min",
            "Rest: easy walk 15-20 min + stretching",
        ]
        increasePlan = [
            "Day type A: Strength (full body) 35-45 min (progressive overload)",
            "Day type B: Easy cardio/walk 25-30 min + recovery",
            "Day type A: Strength 35-45 min (legs + push + pull)",
            "Rest: mobility + 10-15 min walk",
            "Day type C: Strength + short cardio (20 min) + core 10 min",
            "Day type A: Strength 35-45 min + protein shake/meal",
            "Rest: easy walk + sleep focus",
        ]
        maintainPlan = [
            "Day type A: Strength 30-40 min + core 10 min",
            "Day type B: Walk 30 min + mobility",
            "Day type C: Strength 30-40 min (different exercises)",
            "Rest: stretching + easy walk",
            "Day type A: Walk 35 min + light cardio",
            "Day type B: Strength 30-40 min",
            "Rest: recovery day",
        ]

        workoutPattern = reducePlan if action == "reduce" else increasePlan if action == "increase" else maintainPlan

        plan_summary = (
            f"AI-style follow-it plan for {plan_days} days ({action} goal). "
            "Consistency matters more than perfection."
        )
        for day in range(1, plan_days + 1):
            idx = (day - 1) % 7
            plan_schedule.append(
                f"Day {day}: {baseDietLine} Include 25-35g protein per meal; avoid sugary snacks."
            )
            workout_schedule.append(f"Day {day}: {workoutPattern[idx]}")

    # Optional: override the rule-based plan with an LLM (Ollama) when enabled.
    # If the LLM fails or returns invalid JSON, we keep the rule-based plan.
    try:
        if "action" in locals() and action is not None:
            llm_plan = _generate_plan_with_ollama(
                action=action,
                bmi_category=bmi_category or "Unknown",
                bmi=bmi,
                height_cm=payload.height_cm,
                weight_kg=payload.weight_kg,
                plan_days=plan_days,
            )
            if llm_plan:
                plan_summary = llm_plan.get("plan_summary", plan_summary)
                estimated_change_text = llm_plan.get("estimated_change_text", estimated_change_text)
                plan_schedule = llm_plan.get("plan_schedule", plan_schedule)
                workout_schedule = llm_plan.get("workout_schedule", workout_schedule)
    except Exception:
        # Never fail the API if LLM integration has issues.
        pass

    return AgeCheckResponse(
        name=payload.name.strip(),
        years=years,
        months=months,
        days=days,
        hours=total_hours,
        minutes=total_minutes,
        seconds=total_seconds,
        elapsed_at_epoch_ms=int(now_dt.timestamp() * 1000),
        lifetime_years=lifetime_years,
        lifetime_completed_percent=lifetime_completed_percent,
        bmi=bmi,
        bmi_category=bmi_category,
        bmi_advice=bmi_advice,
        diet_advice=diet_advice,
        height_cm=payload.height_cm,
        weight_kg=payload.weight_kg,
        plan_days=plan_days,
        plan_summary=plan_summary,
        estimated_change_text=estimated_change_text,
        plan_schedule=plan_schedule,
        workout_schedule=workout_schedule,
    )
