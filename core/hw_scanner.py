# core/hw_scanner.py — Hardware ONLY: RAM, GPU, Disk
import platform, logging, shutil
from typing import Dict
logger = logging.getLogger(__name__)

class HWScanner:
    def scan(self) -> Dict:
        return {"ram": self._ram(), "gpu": self._gpu(),
                "disk": self._disk(), "os": platform.system()}

    def _ram(self):
        gb = 8.0
        try:
            import psutil
            gb = round(psutil.virtual_memory().total / (1024**3), 1)
        except Exception:
            try:
                if platform.system() == "Linux":
                    with open("/proc/meminfo") as f:
                        for line in f:
                            if "MemTotal" in line:
                                gb = round(int(line.split()[1])/(1024**2), 1); break
                elif platform.system() == "Darwin":
                    import subprocess
                    r = subprocess.run(["sysctl","-n","hw.memsize"], capture_output=True, text=True)
                    gb = round(int(r.stdout.strip())/(1024**3), 1)
            except Exception: pass
        return {"gb": gb, "tier": "high" if gb>=16 else ("mid" if gb>=8 else "low")}

    def _gpu(self):
        try:
            import torch
            if torch.cuda.is_available():
                vram = round(torch.cuda.get_device_properties(0).total_memory/(1024**3),1)
                return {"available":True,"name":torch.cuda.get_device_name(0),"vram_gb":vram,"type":"CUDA"}
        except Exception: pass
        try:
            import subprocess
            r = subprocess.run(["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader"],
                               capture_output=True, text=True, timeout=3)
            if r.returncode==0 and r.stdout.strip():
                parts = r.stdout.strip().split(",")
                vram  = round(int(parts[1].strip().split()[0])/1024,1) if len(parts)>1 else 0
                return {"available":True,"name":parts[0].strip(),"vram_gb":vram,"type":"CUDA"}
        except Exception: pass
        return {"available":False,"name":"CPU only","vram_gb":0,"type":"CPU"}

    def _disk(self):
        try:
            total,used,free = shutil.disk_usage("/")
            return {"free_gb":round(free/(1024**3),1),"total_gb":round(total/(1024**3),1)}
        except Exception:
            return {"free_gb":50,"total_gb":100}

    def recommend_size(self, hw: Dict) -> str:
        ram  = hw["ram"]["gb"]
        vram = hw["gpu"].get("vram_gb",0)
        if ram>=16 or vram>=12: return "large"
        if ram>=8  or vram>=6:  return "medium"
        return "small"
