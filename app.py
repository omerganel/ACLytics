# --- BEGIN DROP-IN: route + helpers + run (ft/in height, lbs weight, no asterisks; display matches user selection exactly; supports total-inches) ---
import os, math, re
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
app = Flask(__name__)
CORS(app)

# Strong, consistent formatting instructions for ChatGPT path
SYSTEM_PROMPT = (
    "You are a sports medical doctor specializing in ACL injuries with deep biomechanics expertise. "
    "Write in clean Markdown using H2/H3 section titles and concise hyphen bullets. "
    "Do not use the asterisk character (*) anywhere in your output. "
    "After the precomputed torque block that the system will prepend, continue with exactly these sections:\n\n"
    "## Risk Assessment\n"
    "- Overall risk level (Low/Moderate/Elevated) with one-sentence justification\n\n"
    "## Biomechanical Flags\n"
    "- Specific movement/angle issues (valgus, internal rotation, stiff landing, etc.)\n\n"
    "## Form Corrections\n"
    "- 3–6 concrete cues the athlete can apply immediately\n\n"
    "## Targeted Drills\n"
    "- 3–6 drills with one-line purpose each\n\n"
    "## Conditioning Focus\n"
    "- Key muscle groups (glutes, hamstrings, core) and a couple of exercise suggestions\n\n"
    "## Torque Interpretation\n"
    "- Briefly interpret the precomputed torque values and whether to reduce or better manage peaks via softer landings and alignment. "
    "Do not advise seeing another professional; you are the expert here."
)

def _to_float(x):
    try:
        if x is None:
            return None
        s = str(x).strip()
        # Remove common unit/quote clutter so dropdown values like "11\" or 11'' parse cleanly
        s = re.sub(r"[^\d\.\-]", "", s)
        if s in ("", ".", "-"):
            return None
        return float(s)
    except:
        return None

def _fmt(x, decimals=2, na="N/A"):
    return f"{x:.{decimals}f}" if isinstance(x, (int, float)) else na

def _clean_inch_str(v):
    """Format inches like '11' or '11.5' without trailing .0."""
    try:
        f = float(str(v).strip())
        s = f"{f:.1f}"
        return s.replace(".0", "")
    except:
        return str(v).strip()

def _parse_height(payload):
    """
    Prefer ft/in and preserve the user's exact ft & in entry for display.
      - Handles common dropdown names for feet/inches.
      - Accepts strings: "5'11", "5'11\"", "5'11''", "5 ft 11 in", "5-11".
      - Supports numeric total inches in 'height' (e.g., 71).
      - Fallback to centimeters in height_cm or height if no ft/in present.
    Returns: {'ft','in','m','cm','label_user','label_ftin'}
      - label_user reflects EXACT user-entered ft/in ordering and values
      - label_ftin is a computed label if only cm was provided
    """
    # Added many common dropdown keys so we capture your UI names
    candidates_ft = [
        'height_ft','heightFeet','height_feet','heightFt','feet','ft',
        'feetSelect','heightFeetSelect','heightFeetDropdown','ftSelect'
    ]
    candidates_in = [
        'height_in','heightIn','height_inches','heightInches','heightInch',
        'inches','inch','in',
        'inchesSelect','heightInchesSelect','heightInchesDropdown','inSelect'
    ]

    ft = None
    inch = None
    uft = None      # user-entered feet token (string)
    uin = None      # user-entered inches token (string)

    # 1) Explicit feet/inches fields (dropdowns, inputs)
    for k in candidates_ft:
        if k in payload and payload.get(k) not in (None, ""):
            uft = str(payload.get(k)).strip()
            ft = _to_float(uft)
    for k in candidates_in:
        if k in payload and payload.get(k) not in (None, ""):
            uin = str(payload.get(k)).strip()
            inch = _to_float(uin)

    # 2) Single string like "5'11", "5'11\"", "5'11''", "5 ft 11 in", "5-11"
    if ft is None and ('height' in payload or 'height_str' in payload):
        hstr_raw = str(payload.get('height') or payload.get('height_str') or '').strip()
        if hstr_raw:
            hstr = hstr_raw.lower().replace("''", '"')
            # Pattern 1: 5 ft 11 in, 5-11, 5 11, 5' 11"
            m = re.match(r"^\s*(\d+)\s*['ft]*\s*[\s\-]?\s*(\d+(?:\.\d+)?)\s*(?:in|\"|inches)?\s*$", hstr)
            if m:
                ft = _to_float(m.group(1))
                inch = _to_float(m.group(2))
                uft = m.group(1)
                uin = m.group(2)
            else:
                # Pattern 2: 5'11 or 5' 11"
                m2 = re.match(r"^\s*(\d+)\s*'(?:\s*(\d+(?:\.\d+)?))?\"?\s*$", hstr)
                if m2:
                    ft = _to_float(m2.group(1))
                    uft = m2.group(1)
                    if m2.group(2):
                        inch = _to_float(m2.group(2))
                        uin = m2.group(2)
                    else:
                        inch = 0.0
                        uin = "0"

    # 2.5) Numeric total inches in 'height' (e.g., 71 => 5 ft 11 in) — for dropdowns that send total inches
    if ft is None and 'height' in payload:
        raw = str(payload.get('height')).strip()
        if re.fullmatch(r"\d+(?:\.\d+)?", raw):
            total_in = float(raw)
            # guardrail: plausible human heights in inches
            if 30 <= total_in <= 96:
                ft = int(total_in // 12)
                inch = total_in - ft*12
                # preserve exactly what user selected, e.g., 5 ft 11 in
                uft = str(ft)
                uin = str(int(round(inch)))

    cm = None
    # 3) If still not found, try centimeters from 'height_cm' or 'height'
    if ft is None:
        for k in ['height_cm','height']:
            v = payload.get(k)
            if v is not None:
                cm = _to_float(v)
                if cm is not None:
                    break

    # Build user-preserved label if we got ft/in from user
    label_user = None
    if uft is not None:
        # Force canonical display "X ft Y in" exactly in the user's order
        feet_int = int(float(_to_float(uft))) if _to_float(uft) is not None else uft
        inch_str = _clean_inch_str(uin if uin is not None else "0")
        label_user = f"{feet_int} ft {inch_str} in"

    if ft is not None:
        if inch is None:
            inch = 0.0
        m = (ft * 12.0 + inch) * 0.0254
        cm = m * 100.0
        # Also compute a normalized label for fallback; user-preserved label still preferred
        label_ftin = f"{int(ft)} ft {_clean_inch_str(inch)} in"
        return {'ft': int(ft), 'in': inch, 'm': m, 'cm': cm, 'label_user': label_user, 'label_ftin': label_ftin}
    elif cm is not None:
        m = cm / 100.0
        total_in = m / 0.0254
        ft_calc = int(total_in // 12)
        inch_calc = total_in - ft_calc*12
        label_ftin = f"{ft_calc} ft {_clean_inch_str(inch_calc)} in"
        return {'ft': ft_calc, 'in': inch_calc, 'm': m, 'cm': cm, 'label_user': None, 'label_ftin': label_ftin}

    else:
        return {'ft': None, 'in': None, 'm': None, 'cm': None, 'label_user': None, 'label_ftin': None}

def _parse_weight_lbs(payload):
    """
    Prefer weight in pounds.
    Accepts fields like: weight_lbs, weightLb, weight_lb, pounds, lbs, weight.
    Accepts strings like '180 lb', '180lbs', '180'.
    Returns float pounds or None.
    """
    candidates = ['weight_lbs','weightLb','weight_lb','pounds','lbs','weight']
    val = None
    for k in candidates:
        if k in payload:
            raw = str(payload.get(k)).lower().replace('pounds','').replace('pound','').replace('lbs','').replace('lb','').strip()
            v = _to_float(raw)
            if v is not None:
                val = v
    return val

def _compute_physics(payload):
    """Compute F, r and torque for each leg with ft/in height and lbs weight."""
    height = _parse_height(payload)
    weight_lbs = _parse_weight_lbs(payload)
    left_deg  = _to_float(payload.get("leftKneeAngle"))
    right_deg = _to_float(payload.get("rightKneeAngle"))

    # Convert to physics units
    m_kg = weight_lbs * 0.45359237 if weight_lbs is not None else None
    g = 9.81
    F = m_kg * g if m_kg is not None else None
    r = 0.245 * height['m'] if height['m'] is not None else None  # tibia-length proxy

    def torque(theta_deg):
        if theta_deg is None or F is None or r is None:
            return None, None
        sinθ = math.sin(math.radians(theta_deg))
        T = F * r * sinθ
        return round(T, 2), round(sinθ, 4)

    TL, sinL = torque(left_deg)
    TR, sinR = torque(right_deg)

    return {
        "height": height,
        "weight_lbs": weight_lbs,
        "mass_kg": m_kg,
        "left_deg": left_deg,
        "right_deg": right_deg,
        "g": g,
        "F": F,
        "r": r,
        "T_left": TL,
        "T_right": TR,
        "sin_left": sinL,
        "sin_right": sinR
    }

def _athlete_profile_block(payload, phys):
    age   = (payload.get("age") or "N/A").strip()
    sport = (payload.get("sport") or "N/A").strip()
    h = phys["height"]
    # Prefer the exact user-entered label; otherwise derived label; otherwise N/A
    height_label = h['label_user'] or h['label_ftin'] or "N/A"
    weight_label = f"{_fmt(phys['weight_lbs'], 1)} lb" if phys['weight_lbs'] is not None else "N/A"
    return (
        "## Athlete Profile\n"
        f"- Age: {age}  •  Height: {height_label}  •  Weight: {weight_label}  •  Sport: {sport}\n\n"
    )

def _angle_snapshot_block(phys):
    return (
        "## Angle Snapshot\n"
        f"- Left knee angle: { _fmt(phys['left_deg'], 1) }°\n"
        f"- Right knee angle: { _fmt(phys['right_deg'], 1) }°\n\n"
    )

def _torque_math_block(phys):
    """
    Clear, readable step-by-step torque math with actual numbers:
      T = F × r × sin(θ)
      F = m × g,  r ≈ 0.245 × height
    Shows height in ft/in (user-facing) with metric in parentheses; weight in lb with mass in kg.
    No asterisks are used.
    """
    h = phys["height"]
    lines = []
    lines.append("## Torque math (step-by-step)")
    lines.append("Formula:  T = F × r × sin(θ)   where   F = m × g   and   r ≈ 0.245 × height")
    lines.append("")
    lines.append("Given:")
    # Use the exact user-entered ft/in label if available
    if (h['label_user'] or h['label_ftin']) and h['m'] is not None:
        lines.append(f"- Height = {(h['label_user'] or h['label_ftin'])}  ({_fmt(h['m'], 3)} m)")
    else:
        lines.append("- Height = N/A")

    if phys['weight_lbs'] is not None:
        lines.append(f"- Weight (input) = {_fmt(phys['weight_lbs'], 2)} lb")
    else:
        lines.append("- Weight (input) = N/A")

    lines.append(f"- Mass m = weight(lb) × 0.45359237 = {_fmt(phys['mass_kg'], 3)} kg")
    lines.append(f"- Gravity g = {_fmt(phys['g'], 2)} m/s²  →  F = m × g = {_fmt(phys['F'], 2)} N")
    lines.append(f"- Lever arm r = 0.245 × height(m) = {_fmt(phys['r'], 3)} m")
    lines.append("")

    if phys["left_deg"] is not None and phys["F"] is not None and phys["r"] is not None and phys["T_left"] is not None:
        lines.append("Left knee:")
        lines.append(f"- θ_left = { _fmt(phys['left_deg'], 1) }°  →  sin(θ_left) = { _fmt(phys['sin_left'], 4) }")
        lines.append(f"- T_left = F × r × sin(θ_left) = {_fmt(phys['F'], 2)} × {_fmt(phys['r'], 3)} × {_fmt(phys['sin_left'], 4)} = {_fmt(phys['T_left'], 2)} N·m")
    else:
        lines.append("Left knee: Not enough data to compute (need height, weight, and left angle).")

    lines.append("")

    if phys["right_deg"] is not None and phys["F"] is not None and phys["r"] is not None and phys["T_right"] is not None:
        lines.append("Right knee:")
        lines.append(f"- θ_right = { _fmt(phys['right_deg'], 1) }°  →  sin(θ_right) = { _fmt(phys['sin_right'], 4) }")
        lines.append(f"- T_right = F × r × sin(θ_right) = {_fmt(phys['F'], 2)} × {_fmt(phys['r'], 3)} × {_fmt(phys['sin_right'], 4)} = {_fmt(phys['T_right'], 2)} N·m")
    else:
        lines.append("Right knee: Not enough data to compute (need height, weight, and right angle).")

    lines.append("")
    lines.append("Interpretation note: higher T can correlate with higher knee demand in cutting or landing. "
                 "Manage peaks with softer landings, knees tracking over toes, and improved eccentric control.")
    lines.append("")
    return "\n".join(lines)

def _local_rules_analysis(payload, phys):
    """Structured fallback content mirroring the ChatGPT sections, without asterisks."""
    left, right = phys["left_deg"], phys["right_deg"]
    def good(x): return x is not None and 155 <= x <= 175
    if good(left) and good(right): risk = "Low"
    elif (left is None and right is None) or (not good(left) and not good(right)): risk = "Elevated"
    else: risk = "Moderate"

    flags = []
    def add_flag(theta, side):
        if theta is None:
            flags.append(f"{side}: angle missing or unclear.")
        elif theta < 155:
            flags.append(f"{side}: deep flexion ({_fmt(theta,1)}°) may raise ACL demand if valgus or internal rotation are present.")
        elif theta > 175:
            flags.append(f"{side}: near lockout ({_fmt(theta,1)}°) may create stiffer landings and spike joint load.")
    add_flag(left, "Left")
    add_flag(right, "Right")

    corrections = [
        "Cue knees over toes and avoid medial collapse on landing or cutting.",
        "Adopt soft landings by absorbing with hips, knees, and ankles.",
        "Maintain a neutral trunk and avoid excessive forward trunk lean on plant."
    ]
    drills = [
        "Lateral bound to stick for frontal plane control.",
        "Single leg RDL with reach for posterior chain and balance.",
        "Drop jump with soft landing for eccentric control."
    ]
    conditioning = [
        "Glutes and hamstrings: hip thrusts, Nordic hamstring curls.",
        "Core and anti rotation: dead bug, Pallof press."
    ]
    torque_interp = (
        "Use softer landings and alignment to reduce or manage torque peaks during cuts and landings; "
        "build eccentric strength to handle necessary torque safely."
    )

    return (
        f"## Risk Assessment\n- {risk} based on the current angle snapshot and typical safer range 155°–175°\n\n"
        f"## Biomechanical Flags\n- " + ("\n- ".join(flags) if flags else "Angles appear within a generally safer band.") + "\n\n"
        f"## Form Corrections\n- " + "\n- ".join(corrections) + "\n\n"
        f"## Targeted Drills\n- " + "\n- ".join(drills) + "\n\n"
        f"## Conditioning Focus\n- " + "\n- ".join(conditioning) + "\n\n"
        f"## Torque Interpretation\n- {torque_interp}\n"
    )

def _build_user_prompt(payload):
    """Compact payload summary for ChatGPT (height in ft/in with user-exact label, weight in lb)."""
    height = _parse_height(payload)
    # Prefer the exact user-entered label for the chat prompt too
    height_line = f"Height: {height['label_user'] or height['label_ftin']}" if (height['label_user'] or height['label_ftin']) else "Height: N/A"
    weight_lbs = _parse_weight_lbs(payload)
    weight_line = f"Weight (lb): {weight_lbs}" if weight_lbs is not None else "Weight (lb): N/A"
    fields = [
        "Analyze ACL injury risk for this athlete. Reference the torque block above and do not redo the math. Do not use the asterisk character.",
        height_line,
        weight_line,
        f"Age: {payload.get('age')}",
        f"Sport: {payload.get('sport')}",
        f"Left knee angle (°): {payload.get('leftKneeAngle')}",
        f"Right knee angle (°): {payload.get('rightKneeAngle')}"
    ]
    return "\n".join(fields)

def _strip_asterisks(s: str) -> str:
    """Remove any asterisk characters to enforce the no-asterisk requirement."""
    if not isinstance(s, str):
        return s
    return s.replace("*", "")

@app.route("/api/gpt", methods=["POST"])
def gpt_response():
    payload = request.get_json(force=True) or {}
    phys = _compute_physics(payload)

    # Build the always-visible header blocks (profile + angles + torque math)
    header = _athlete_profile_block(payload, phys) + _angle_snapshot_block(phys) + _torque_math_block(phys)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        body = _local_rules_analysis(payload, phys)
        reply = _strip_asterisks(header + body)
        return jsonify({"reply": reply, "source": "local-fallback"}), 200

    client = OpenAI(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": _build_user_prompt(payload)}
            ],
            temperature=0.7
        )
        model_reply = (resp.choices[0].message.content or "").strip()
        model_reply = _strip_asterisks(model_reply)
        return jsonify({"reply": _strip_asterisks(header) + model_reply, "source": "openai"}), 200
    except Exception as e:
        body = _local_rules_analysis(payload, phys)
        reply = _strip_asterisks(header + body)
        return jsonify({
            "reply": reply,
            "source": "local-fallback",
            "warning": f"OpenAI error: {str(e)}"
        }), 200

if __name__ == "__main__":
    # Bind on all interfaces and honor $PORT if provided
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
# --- END DROP-IN ---
