import sys
from streamlit.testing.v1 import AppTest


def test_streamlit_app_boots():
    """Verify SuryaJal Streamlit entrypoint boots cleanly without runtime exceptions."""
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert not at.exception, f"AppTest exceptions: {[exc.value for exc in at.exception]}"


if __name__ == "__main__":
    print("Running AppTest on app.py...")
    try:
        test_streamlit_app_boots()
        print("AppTest ran cleanly with 0 exceptions!")
        sys.exit(0)
    except Exception as e:
        print(f"Test crashed with error: {e}")
        sys.exit(1)
