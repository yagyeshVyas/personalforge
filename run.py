#!/usr/bin/env python3
"""PersonalForge v5 — Run: python run.py"""
import subprocess, sys, os, webbrowser, time, threading

def open_browser():
    time.sleep(1.5)
    webbrowser.open("http://localhost:5000")

print("""
╔══════════════════════════════════════════════════╗
║          PersonalForge v5                    ║
║   Build Your Own AI · Free · No Coding          ║
╠══════════════════════════════════════════════════╣
║   Opening: http://localhost:5000                ║
╚══════════════════════════════════════════════════╝
""")

threading.Thread(target=open_browser, daemon=True).start()

os.chdir(os.path.dirname(os.path.abspath(__file__)))
subprocess.run([sys.executable, "server.py"])
