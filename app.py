# Entry point for Streamlit Cloud deployment
# Supports both 'streamlit_app.py' and 'app.py' entry-point settings
import os
import sys
import runpy

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

target_script = os.path.join(CURRENT_DIR, "streamlit_app.py")
if os.path.exists(target_script):
    runpy.run_path(target_script, run_name="__main__")
else:
    raise FileNotFoundError(f"Cannot locate streamlit_app.py in {CURRENT_DIR}")
