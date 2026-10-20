import unittest
from unittest.mock import patch

from dashboard.alert_controls import (
    PREFERENCE_DEFAULTS,
    _browser_controls_html,
    announcement_text,
    initialize_alert_state,
    register_persisted_incident,
    severity_visual_style,
)
from dashboard.camera_sources import camera_connection_state
from dashboard import theme


class AlertPreferenceTests(unittest.TestCase):
    def test_dark_and_light_theme_tokens_are_applied(self):
        with patch.object(theme.st, "markdown") as render_css:
            theme.apply_theme("Dark")
            dark_css = render_css.call_args.args[0]
            theme.apply_theme("Light")
            light_css = render_css.call_args.args[0]
        self.assertIn("color-scheme: dark", dark_css)
        self.assertIn("--ad-bg:#08121d", dark_css)
        self.assertIn("color-scheme: light", light_css)
        self.assertIn("--ad-bg:#f3f6f8", light_css)

    def test_session_preferences_survive_rerun_initialization(self):
        state = {}
        initialize_alert_state(state)
        state["theme_mode"] = "Light"
        state["browser_notifications"] = True
        initialize_alert_state(state)
        self.assertEqual(state["theme_mode"], "Light")
        self.assertTrue(state["browser_notifications"])
        self.assertEqual(state["voice_language"], PREFERENCE_DEFAULTS["voice_language"])

    def test_duplicate_persisted_id_is_registered_once(self):
        state = {}
        incident = {"id": 27, "event_type": "Crowd Density", "severity": "Medium", "source": "Browser webcam"}
        self.assertTrue(register_persisted_incident(state, incident))
        self.assertFalse(register_persisted_incident(state, incident))
        self.assertEqual([item["id"] for item in state["pending_incident_alerts"]], [27])

    def test_startup_has_no_historical_alerts_to_dispatch(self):
        state = {}
        initialize_alert_state(state)
        self.assertEqual(state["pending_incident_alerts"], [])
        self.assertEqual(state["alert_registered_ids"], [])

    def test_severity_has_distinct_visual_styles(self):
        self.assertEqual(severity_visual_style("High"), "high")
        self.assertEqual(severity_visual_style("Medium"), "medium")
        self.assertEqual(severity_visual_style("Low"), "low")

    def test_english_and_hinglish_announcement_copy_is_heuristic_safe(self):
        event = "Possible Fall (posture heuristic)"
        self.assertEqual(
            announcement_text(event, "High", "English"),
            "High priority alert. Possible fall detected. Please review camera evidence.",
        )
        self.assertEqual(
            announcement_text(event, "High", "Hinglish"),
            "High priority alert. Possible fall detect hua hai. Kripya camera evidence check karein.",
        )
        self.assertIn("Possible altercation", announcement_text("Possible Physical Altercation (motion heuristic)", "Medium"))

    def test_browser_api_absence_has_guarded_fallbacks(self):
        prefs = {"browser_notifications": True, "sound_alerts": True, "voice_alerts": True, "voice_language": "English"}
        html = _browser_controls_html([], prefs)
        self.assertIn("'Notification' in window", html)
        self.assertIn("'speechSynthesis' in window", html)
        self.assertIn("AudioContext||window.webkitAudioContext", html)
        self.assertIn("Notifications are blocked by browser settings", html)

    def test_camera_status_requires_real_stream_and_recent_inference(self):
        now = 1000.0
        state = {"webcam_stream_active": False, "last_detection_metrics": {"source": "Browser webcam", "updated_at": now}}
        self.assertEqual(camera_connection_state(state, now)["browser_status"], "Not connected")
        self.assertFalse(camera_connection_state(state, now)["browser_inference"])
        state["webcam_stream_active"] = True
        self.assertTrue(camera_connection_state(state, now)["browser_inference"])
        state["last_detection_metrics"]["updated_at"] = now - 10
        self.assertEqual(camera_connection_state(state, now)["browser_status"], "Connected")
        self.assertFalse(camera_connection_state(state, now)["browser_inference"])


if __name__ == "__main__":
    unittest.main()
