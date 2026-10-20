"""Session-scoped alert preferences and browser-native incident cues."""

from __future__ import annotations

import json
import logging

import streamlit as st

from dashboard.live_feed import format_saved_timestamp_local

logger = logging.getLogger(__name__)

PREFERENCE_DEFAULTS = {
    "theme_mode": "Dark",
    "visual_high_alerts": True,
    "browser_notifications": False,
    "sound_alerts": False,
    "voice_alerts": False,
    "voice_language": "English",
}


def initialize_alert_state(state) -> None:
    for key, value in PREFERENCE_DEFAULTS.items():
        if key not in state:
            state[key] = value
    state.setdefault("alert_registered_ids", [])
    state.setdefault("pending_incident_alerts", [])
    state.setdefault("active_visual_alert", None)
    state.setdefault("acknowledged_alert_ids", [])


def register_persisted_incident(state, incident: dict) -> bool:
    """Queue only a newly persisted ID from this live Streamlit session."""
    initialize_alert_state(state)
    incident_id = incident.get("id") if isinstance(incident, dict) else None
    if incident_id is None:
        return False
    try:
        incident_id = int(incident_id)
    except (TypeError, ValueError, OverflowError):
        return False
    registered = state["alert_registered_ids"]
    if incident_id in registered:
        return False
    state["alert_registered_ids"] = (registered + [incident_id])[-500:]
    state["pending_incident_alerts"].append({
        "id": incident_id,
        "event_type": str(incident.get("event_type") or "Unknown event"),
        "severity": str(incident.get("severity") or "Low"),
        "source": str(incident.get("source") or ""),
    })
    return True


def severity_visual_style(severity: str) -> str:
    return {"High": "high", "Medium": "medium", "Low": "low"}.get(severity, "low")


def announcement_text(event_type: str, severity: str, language: str = "English") -> str:
    kind = str(event_type or "incident")
    lowered = kind.casefold()
    if "fall" in lowered:
        event_en, event_hi = "Possible fall detected", "Possible fall detect hua hai"
    elif "altercation" in lowered or "physical" in lowered:
        event_en, event_hi = "Possible altercation detected", "Possible altercation ka signal mila hai"
    elif "crowd" in lowered:
        event_en, event_hi = "High crowd density detected", "High crowd density detect hui hai"
    elif "bag" in lowered:
        event_en, event_hi = "Bag detection reported", "Bag detection ka alert mila hai"
    else:
        event_en, event_hi = f"{kind} reported", f"{kind} ka alert mila hai"
    priority = {"High": "High priority", "Medium": "Medium priority", "Low": "Low priority"}.get(severity, "Priority")
    if language == "Hinglish":
        return f"{priority} alert. {event_hi}. Kripya camera evidence check karein."
    return f"{priority} alert. {event_en}. Please review camera evidence."


def _friendly_source(value: str) -> str | None:
    source = (value or "").strip()
    if source == "Browser webcam":
        return "Browser webcam"
    if source.startswith("Webcam ") and "local camera" in source:
        return None
    if source.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
        return None
    normalized = source.replace("\\", "/")
    if "/" in normalized:
        return None
    return source or None


def _resolve_incidents(pending: list[dict]) -> list[dict]:
    if not pending:
        return []
    try:
        from database.database import fetch_alerts_by_ids

        rows = {int(row["id"]): dict(row) for row in fetch_alerts_by_ids(item["id"] for item in pending)}
    except Exception:
        logger.exception("Could not load newly saved incident details for alert delivery")
        return None
    resolved = []
    for item in pending:
        row = rows.get(item["id"])
        if row is None:
            continue
        resolved.append({
            "id": item["id"],
            "event_type": str(row.get("event_type") or item["event_type"]),
            "severity": str(row.get("severity") or item["severity"]),
            "timestamp": format_saved_timestamp_local(row.get("timestamp")),
            "source": _friendly_source(str(row.get("source") or item["source"])),
        })
    return resolved


def _browser_controls_html(events: list[dict], prefs: dict) -> str:
    safe_events = json.dumps(events, ensure_ascii=True).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    flags = json.dumps({
        "notifications": bool(prefs["browser_notifications"]),
        "sound": bool(prefs["sound_alerts"]),
        "voice": bool(prefs["voice_alerts"]),
        "language": prefs["voice_language"],
    }, ensure_ascii=True)
    template = r'''<meta charset="utf-8"><style>
      .ad-alert-controls{font:13px system-ui,sans-serif;color:var(--ad-body,#314452);background:transparent}
      .row{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
      .ad-alert-controls button{border:1px solid var(--ad-border,#9aabb7);border-radius:7px;background:var(--ad-raised,#f4f8fa);color:var(--ad-text,#18303d);padding:7px 10px;font-weight:600;cursor:pointer}
      .ad-alert-controls button:focus-visible{outline:3px solid var(--ad-accent,#087e95);outline-offset:2px}
      .ad-alert-controls #status{color:var(--ad-muted,#536a79);font-size:12px}
    </style><div class="ad-alert-controls"><div class="row" id="controls"></div><div id="status" role="status" aria-live="polite"></div></div>
    <script>
    (()=>{
      const events=__EVENTS__; const prefs=__PREFS__; const root=document.getElementById('controls');
      const status=document.getElementById('status');
      const say=(text)=>{status.textContent=text;};
      const addButton=(label,fn)=>{const b=document.createElement('button');b.type='button';b.textContent=label;b.onclick=fn;root.appendChild(b);};
      const getSeen=(name)=>{try{return new Set(JSON.parse(localStorage.getItem(name)||'[]'));}catch(_){return new Set();}};
      const saveSeen=(name,set)=>{try{localStorage.setItem(name,JSON.stringify([...set].slice(-300)));}catch(_){}};
      if(prefs.notifications){
        if(!('Notification' in window)){say('This browser does not support desktop notifications.');}
        else {
          addButton('Enable browser notifications',async()=>{
            try{const result=await Notification.requestPermission();say(result==='granted'?'Browser notification permission granted.':`Browser notification permission: ${result}.`);}
            catch(_){say('Browser notification permission could not be requested here. Use a secure top-level page (HTTPS or localhost).');}
          });
          if(Notification.permission==='denied')say('Notifications are blocked by browser settings.');
          else if(Notification.permission==='default')say('Click Enable browser notifications to grant permission.');
        }
      }
      if(prefs.sound){
        addButton('Enable alert sound',()=>{try{const C=window.AudioContext||window.webkitAudioContext;if(!C){say('This browser does not support alert sound.');return;}const ctx=new C();ctx.resume().then(()=>{localStorage.setItem('ad_sound_unlocked','1');say('Alert sound enabled for this browser session.');ctx.close();}).catch(()=>say('The browser blocked audio. Click Enable alert sound again.'));}catch(_){say('Alert sound is unavailable in this browser.');}});
      }
      if(prefs.voice){
        addButton('Enable voice alerts',()=>{try{if(!('speechSynthesis' in window)){say('This browser does not support spoken alerts.');return;}speechSynthesis.cancel();localStorage.setItem('ad_voice_unlocked','1');localStorage.setItem('ad_voice_muted','0');say('Browser speech is enabled.');}catch(_){say('Browser speech could not be enabled.');}});
        addButton('Mute / stop voice',()=>{try{if('speechSynthesis' in window)speechSynthesis.cancel();localStorage.setItem('ad_voice_muted','1');say('Voice alerts muted. Use Enable voice alerts to resume.');}catch(_){say('Voice could not be stopped in this browser.');}});
      }
      for(const item of events){
        const id=String(item.id);
        if(prefs.notifications){
          const seen=getSeen('ad_notice_ids');
          if(!seen.has(id)){
            seen.add(id); saveSeen('ad_notice_ids',seen);
            if('Notification' in window && Notification.permission==='granted'){
              try{const body=[item.event_type,item.severity+' severity',item.timestamp,'Incident #'+id].join(' | ');new Notification('ANTA DRISHTI alert',{body,tag:'antadrishti-'+id});}
              catch(_){say('The browser could not display this notification.');}
            }
          }
        }
        if(prefs.sound && item.severity==='High'){
          const seen=getSeen('ad_sound_ids');
          if(!seen.has(id)){
            seen.add(id); saveSeen('ad_sound_ids',seen);
            try{if(localStorage.getItem('ad_sound_unlocked')==='1'){const C=window.AudioContext||window.webkitAudioContext;if(C){const ctx=new C();ctx.resume().then(()=>{const osc=ctx.createOscillator(),gain=ctx.createGain();osc.type='sine';osc.frequency.value=740;gain.gain.setValueAtTime(.0001,ctx.currentTime);gain.gain.exponentialRampToValueAtTime(.12,ctx.currentTime+.02);gain.gain.exponentialRampToValueAtTime(.0001,ctx.currentTime+.22);osc.connect(gain);gain.connect(ctx.destination);osc.start();osc.stop(ctx.currentTime+.24);osc.onended=()=>ctx.close();}).catch(()=>say('The browser blocked the alert tone.'));}}}catch(_){say('The alert tone is unavailable.');}
          }
        }
        if(prefs.voice){
          const seen=getSeen('ad_voice_ids');
          if(!seen.has(id)){
            seen.add(id); saveSeen('ad_voice_ids',seen);
            try{
              if('speechSynthesis' in window && localStorage.getItem('ad_voice_unlocked')==='1' && localStorage.getItem('ad_voice_muted')!=='1'){
                const last=Number(localStorage.getItem('ad_voice_last_at')||0);const now=Date.now();
                const availableVoices=speechSynthesis.getVoices();
                if(!availableVoices.length){say('No browser speech voice is currently available.');}
                else if(!speechSynthesis.speaking && now-last>=5000){const utterance=new SpeechSynthesisUtterance(item.announcement);utterance.lang=prefs.language==='Hinglish'?'en-IN':'en-US';utterance.rate=.92;speechSynthesis.speak(utterance);localStorage.setItem('ad_voice_last_at',String(now));}
              } else if(!('speechSynthesis' in window)){say('This browser does not support spoken alerts.');}
            }catch(_){say('Spoken alert could not be played in this browser.');}
          }
        }
      }
      if(events.length && !status.textContent) say('New incident alert preferences were applied.');
    })();
    </script>'''
    return template.replace("__EVENTS__", safe_events).replace("__PREFS__", flags)


def render_new_incident_alerts() -> None:
    """Render new persisted session events once; never scan old history for alerts."""
    state = st.session_state
    initialize_alert_state(state)
    pending = list(state.get("pending_incident_alerts", []))
    state["pending_incident_alerts"] = []
    incidents = _resolve_incidents(pending)
    if incidents is None:
        state["pending_incident_alerts"] = pending
        incidents = []
    if incidents:
        for item in incidents:
            if item["severity"] == "High" and state["visual_high_alerts"]:
                state["active_visual_alert"] = item
            elif state["visual_high_alerts"] and item["severity"] == "Medium":
                st.toast(f"Medium severity incident #{item['id']}: {item['event_type']}", icon="⚠️")
            elif state["visual_high_alerts"]:
                st.toast(f"Low severity incident #{item['id']}: {item['event_type']}", icon="ℹ️")

    active = state.get("active_visual_alert")
    if not state["visual_high_alerts"]:
        state["active_visual_alert"] = None
        active = None
    if state["visual_high_alerts"] and active and int(active["id"]) not in state["acknowledged_alert_ids"]:
        st.error(
            f"HIGH PRIORITY | {active['event_type']} | Incident #{active['id']} | "
            f"{active['timestamp']} | Source: {active['source'] or 'Unavailable'}"
        )
        if "heuristic" in active["event_type"].casefold() or "possible" in active["event_type"].casefold():
            st.caption("Possible event only; this signal requires human review and is not a confirmed emergency.")
        if st.button("Acknowledge alert", key=f"acknowledge_incident_{active['id']}"):
            state["acknowledged_alert_ids"] = (state["acknowledged_alert_ids"] + [int(active["id"])])[-500:]
            state["active_visual_alert"] = None

    prefs = {key: state[key] for key in PREFERENCE_DEFAULTS if key != "theme_mode"}
    enabled = prefs["browser_notifications"] or prefs["sound_alerts"] or prefs["voice_alerts"]
    if not enabled:
        return
    event_payload = []
    for item in incidents:
        event_payload.append({
            **item,
            "announcement": announcement_text(item["event_type"], item["severity"], prefs["voice_language"]),
        })
    try:
        st.html(_browser_controls_html(event_payload, prefs), unsafe_allow_javascript=True)
    except Exception:
        logger.exception("Browser alert controls could not be rendered")
        st.caption("Browser notification, sound, and voice controls are unavailable in this browser.")


def render_alert_preferences() -> None:
    initialize_alert_state(st.session_state)
    st.subheader("Alert preferences")
    st.caption("These operator preferences are session-only and reset when this Streamlit session ends.")
    left, right = st.columns(2)
    with left:
        st.toggle("Visual high-priority alerts", key="visual_high_alerts")
        st.toggle("Browser notifications", key="browser_notifications")
        st.toggle("Sound alerts (high severity)", key="sound_alerts")
    with right:
        st.toggle("Voice announcements", key="voice_alerts")
        st.selectbox("Announcement language", ["English", "Hinglish"], key="voice_language")
        st.caption("Notifications require browser permission. Sound and voice need a click inside the browser controls to unlock playback.")
    st.caption("Desktop notifications work only while this page is open on a supported secure browser origin; no background push service is configured.")
