import os, json, time, threading, subprocess
from datetime import datetime
from pathlib import Path
from flask import Flask, jsonify, request, render_template, Response
import queue

app = Flask(__name__)

DATA_ROOT  = Path(os.environ.get("SYNAPSE_DATA_ROOT", str(Path.home() / "data")))
ENV_PYTHON = os.environ.get("SYNAPSE_PYTHON", "/cis/home/pwu60/my_env/bin/python")
LOG_QUEUE  = queue.Queue(maxsize=500)

pipeline_state = {
    "stages": {
        "registration": {"name":"Registration","tool":"ITK-Elastix","status":"idle","progress":0,"message":"Ready","pid":None},
        "restoration":  {"name":"Restoration","tool":"XTC · 3D U-Net","status":"idle","progress":0,"message":"Ready","pid":None},
        "segmentation": {"name":"Segmentation","tool":"Watershed · Python","status":"idle","progress":0,"message":"Ready","pid":None},
        "tracking":     {"name":"Tracking","tool":"Ilastik / LAP","status":"idle","progress":0,"message":"Ready","pid":None},
    },
    "results": {"sensitivity":None,"precision":None,"snr_improvement":None,"objects_detected":None},
    "jobs": [],
    "system_logs": [],
}
_lock = threading.Lock()

def log(level, stage, msg):
    entry = {"time": datetime.now().strftime("%H:%M:%S"), "level": level.upper(), "stage": stage.upper(), "msg": msg}
    with _lock:
        pipeline_state["system_logs"].append(entry)
        if len(pipeline_state["system_logs"]) > 300:
            pipeline_state["system_logs"].pop(0)
    try:
        LOG_QUEUE.put_nowait(entry)
    except queue.Full:
        pass

def _update(key, **kw):
    with _lock:
        pipeline_state["stages"][key].update(kw)

def _run_stage(stage_key, params):
    _update(stage_key, status="running", progress=0, message="Starting…")
    log("INFO", stage_key, f"Job started")
    cmd = [ENV_PYTHON, "-m", f"pipeline_stages.{stage_key}"] + [f"--{k.replace('_','-')}={v}" for k,v in params.items() if v != "" and v is not None]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(Path.home()/"synapsepipe"))
        _update(stage_key, pid=proc.pid)
        for line in proc.stdout:
            line = line.strip()
            if not line: continue
            log("INFO", stage_key, line)
            if line.startswith("PROGRESS:"):
                pct = int(line.split(":")[1])
                _update(stage_key, progress=pct, message=f"{pct}% complete")
            if line.startswith("SNR:"):
                with _lock: pipeline_state["results"]["snr_improvement"] = float(line.split(":")[1])
            if line.startswith("RESULT:"):
                parts = dict(kv.split("=") for kv in line[7:].split(","))
                with _lock:
                    if "objects"     in parts: pipeline_state["results"]["objects_detected"] = int(parts["objects"])
                    if "sensitivity" in parts: pipeline_state["results"]["sensitivity"]       = float(parts["sensitivity"])
                    if "precision"   in parts: pipeline_state["results"]["precision"]         = float(parts["precision"])
        proc.wait()
        if proc.returncode == 0:
            _update(stage_key, status="done", progress=100, message="Complete ✓", pid=None)
            log("OK", stage_key, "Stage complete ✓")
        else:
            raise RuntimeError(f"Exited with code {proc.returncode}")
    except Exception as e:
        _update(stage_key, status="error", message=str(e), pid=None)
        log("ERROR", stage_key, str(e))

@app.route("/")
def index(): return render_template("dashboard.html")

@app.route("/api/status")
def api_status():
    with _lock: return jsonify(pipeline_state)

@app.route("/api/run/<key>", methods=["POST"])
def api_run(key):
    if key not in pipeline_state["stages"]: return jsonify({"error":"Unknown stage"}), 404
    if pipeline_state["stages"][key]["status"] == "running": return jsonify({"error":"Already running"}), 409
    params = request.get_json(silent=True) or {}
    job = {"id": f"#{int(time.time())%10000}", "stage": key, "started": datetime.now().isoformat(), "status":"running"}
    with _lock:
        pipeline_state["jobs"].insert(0, job)
    threading.Thread(target=_run_stage, args=(key, params), daemon=True).start()
    log("INFO", key, f"Job {job['id']} submitted")
    return jsonify({"ok": True, "job": job})

@app.route("/api/stop/<key>", methods=["POST"])
def api_stop(key):
    import signal
    with _lock: pid = pipeline_state["stages"].get(key, {}).get("pid")
    if not pid: return jsonify({"error":"No running process"}), 404
    try:
        os.kill(pid, signal.SIGTERM)
        _update(key, status="idle", message="Stopped by user", pid=None)
        log("WARN", key, f"Process {pid} stopped")
        return jsonify({"ok": True})
    except ProcessLookupError:
        return jsonify({"error":"Process already finished"}), 404

@app.route("/api/logs/stream")
def api_logs_stream():
    def stream():
        while True:
            try:
                entry = LOG_QUEUE.get(timeout=25)
                yield f"data: {json.dumps(entry)}\n\n"
            except queue.Empty:
                yield 'data: {"ping":true}\n\n'
    return Response(stream(), mimetype="text/event-stream", headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

@app.route("/api/logs")
def api_logs():
    with _lock: return jsonify(pipeline_state["system_logs"][-100:])

if __name__ == "__main__":
    log("OK","SYS","SynapsePipe starting…")
    log("INFO","SYS",f"Data root: {DATA_ROOT}")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)