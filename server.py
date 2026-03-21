# server.py — PersonalForge v10
import os, sys, json, threading
from flask import Flask, render_template, request, jsonify, send_file

sys.path.append(os.path.dirname(__file__))

from core.file_loader    import FileLoader, SOURCE_TYPES
from core.pair_generator import PairGenerator, MODES
from core.data_cleaner   import DataCleaner
from core.hw_scanner     import HWScanner
from core.url_fetcher    import URLFetcher
from core.remote_fetcher import RemoteFetcher
from core.hf_streamer    import HFStreamer
from core.hf_registry    import (get_all as hf_reg_all, search as hf_reg_search,
                                  get_best_model, get_sample_guide)
from core.model_matcher  import ModelMatcher
from core.model_resolver import ModelResolver, POPULAR_MODELS
from core.web_collector  import WebCollector
from colab.notebook_generator import generate as gen_notebook

app      = Flask(__name__)
loader   = FileLoader()
cleaner  = DataCleaner()
genpairs = PairGenerator()
hw       = HWScanner()
uf       = URLFetcher()
rf       = RemoteFetcher()
hfs      = HFStreamer()
matcher  = ModelMatcher()
resolver = ModelResolver()
webcol   = WebCollector()

DATA_DIR   = "data"
OUTPUT_DIR = "output"
PAIRS_FILE = "output/training_pairs.jsonl"
os.makedirs(DATA_DIR,   exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

state = {
    "hw": None, "hf_token": None,
    "chunks": [], "clean_chunks": [], "clean_stats": {}, "quality": {},
    "pairs": [], "sel_model": None, "sel_mode": "factual", "sel_cat": "general",
    "web_topics": [], "nb_path": None,
    "progress": {"step":"","current":0,"total":0,"pairs":0},
}

@app.route("/")
def index():
    return render_template("index.html")

# ── HARDWARE ───────────────────────────────────────────────────────────────────
@app.route("/api/hw_scan", methods=["POST"])
def hw_scan():
    info = hw.scan(); info["rec"] = hw.recommend_size(info)
    state["hw"] = info; return jsonify(info)

@app.route("/api/hw_info")
def hw_info():
    if not state["hw"]:
        info = hw.scan(); info["rec"] = hw.recommend_size(info); state["hw"] = info
    return jsonify(state["hw"])

# ── UPLOAD ─────────────────────────────────────────────────────────────────────
@app.route("/api/upload", methods=["POST"])
def upload():
    files = request.files.getlist("files[]")
    types = request.form.getlist("types[]")
    if not files: return jsonify({"error":"No files"}), 400
    pairs = [(loader.save_bytes(f.filename, f.read(), DATA_DIR),
              SOURCE_TYPES.get(t,"document")) for f,t in zip(files,types) if f.filename]
    chunks,_ = loader.load_many(pairs)
    state["chunks"] += chunks
    return jsonify({"success":True,"added":len(chunks),"total":len(state["chunks"])})

# ── URL/REMOTE ─────────────────────────────────────────────────────────────────
def _is_hf_dataset(url):
    return "huggingface.co/datasets/" in url

def _hf_redirect(url):
    ds = url.replace("https://huggingface.co/datasets/","").split("/")
    return jsonify({"hf_redirect":True,"dataset_id":"/".join(ds[:2])})

@app.route("/api/fetch_url", methods=["POST"])
def fetch_url():
    url = (request.json or {}).get("url","").strip()
    if not url: return jsonify({"error":"No URL"}),400
    if _is_hf_dataset(url): return _hf_redirect(url)
    state["progress"] = {"step":f"Fetching...","current":0,"total":1,"pairs":0}
    def run():
        r = uf.fetch(url)
        if r.get("text") and len(r["text"])>100:
            state["chunks"].append({"text":r["text"],"source":r.get("title",url)[:60],"source_type":r.get("type","web")})
            state["progress"] = {"step":"done","current":1,"total":1,"pairs":r.get("words",0)}
        else: state["progress"] = {"step":"error","current":0,"total":1,"pairs":0}
    threading.Thread(target=run,daemon=True).start()
    return jsonify({"success":True})

@app.route("/api/fetch_remote", methods=["POST"])
def fetch_remote():
    url = (request.json or {}).get("url","").strip()
    if not url: return jsonify({"error":"No URL"}),400
    if _is_hf_dataset(url): return _hf_redirect(url)
    state["progress"] = {"step":"Fetching remote...","current":0,"total":1,"pairs":0}
    def run():
        try:
            r = rf.fetch(url)
            if r.get("text") and len(r["text"])>100:
                state["chunks"].append({"text":r["text"][:50000],"source":r.get("filename","remote"),"source_type":"document"})
                state["progress"] = {"step":"done","current":1,"total":1,"pairs":r.get("words",0)}
            else: state["progress"] = {"step":"error","current":0,"total":1,"pairs":0}
        except Exception as e:
            state["progress"] = {"step":f"error: {str(e)[:60]}","current":0,"total":1,"pairs":0}
    threading.Thread(target=run,daemon=True).start()
    return jsonify({"success":True})

# ── HF TOKEN ───────────────────────────────────────────────────────────────────
@app.route("/api/hf_set_token", methods=["POST"])
def hf_set_token():
    token = (request.json or {}).get("token","").strip()
    if not token:
        state["hf_token"] = None
        return jsonify({"success":True,"message":"Token cleared"})
    result = hfs.validate_token(token)
    if result["valid"]:
        state["hf_token"] = token
        return jsonify({"success":True,"username":result.get("username","HF User"),"message":"Token saved"})
    return jsonify({"success":False,"error":result.get("error","Invalid token")}),401

@app.route("/api/hf_token_status")
def hf_token_status():
    return jsonify({"logged_in":bool(state.get("hf_token")),"username":"HF User" if state.get("hf_token") else ""})

@app.route("/api/hf_check_dataset")
def hf_check_dataset():
    ds = request.args.get("dataset_id","").strip()
    if not ds: return jsonify({"error":"No dataset ID"}),400
    return jsonify(hfs.check_dataset_access(ds, state.get("hf_token")))

# ── HF DATASETS ────────────────────────────────────────────────────────────────
@app.route("/api/hf_categories")
def hf_categories():
    return jsonify([
        {"id":cat_id,"name":cat["name"],"icon":cat["icon"],
         "datasets":[{"id":ds["id"],"name":ds["name"],"desc":ds["desc"],"size":ds["size"],
                      "config":ds.get("config","default"),"split":ds.get("split","train"),
                      "fields":ds.get("fields",{"text":"text"}),"gated":ds.get("gated",False),
                      "star":ds.get("star",False),"token_url":ds.get("token_url",""),
                      "languages":ds.get("languages",[]),"configs_available":ds.get("configs_available",[]),
                      "max_recommended":ds.get("max_recommended",100000)}
                     for ds in cat["datasets"]]}
        for cat_id,cat in hf_reg_all().items()
    ])

@app.route("/api/hf_search")
def hf_search():
    q = request.args.get("q","").strip()
    return jsonify(hf_reg_search(q)[:10] if q else [])

@app.route("/api/hf_stream", methods=["POST"])
def hf_stream():
    d = request.json or {}
    ds_id = d.get("dataset_id",""); n = int(d.get("n_samples",200000))
    if not ds_id: return jsonify({"error":"No dataset"}),400
    state["progress"] = {"step":f"Streaming...","current":0,"total":n,"pairs":0}
    def run():
        try:
            def cb(f,t,ds): state["progress"]={"step":f"Streamed {f:,}/{t:,}","current":f,"total":t,"pairs":0}
            chunks = hfs.stream(dataset_id=ds_id,config=d.get("config"),split=d.get("split","train"),
                                fields=d.get("fields",{"text":"text"}),n_samples=n,
                                lang_filter=d.get("language"),hf_token=state.get("hf_token"),progress_callback=cb)
            state["chunks"]+=chunks
            state["progress"]={"step":"done","current":n,"total":n,"pairs":len(chunks)}
        except Exception as e:
            state["progress"]={"step":f"error: {str(e)[:60]}","current":0,"total":1,"pairs":0}
    threading.Thread(target=run,daemon=True).start()
    return jsonify({"success":True})

@app.route("/api/hf_api_stream", methods=["POST"])
def hf_api_stream():
    d = request.json or {}
    ds_id = d.get("dataset_id","").strip(); n = int(d.get("n_samples",200000))
    if not ds_id: return jsonify({"error":"No dataset"}),400
    state["progress"] = {"step":f"API streaming...","current":0,"total":n,"pairs":0}
    def run():
        try:
            def cb(f,t,ds): state["progress"]={"step":f"Fetched {f:,}/{t:,}","current":f,"total":t,"pairs":0}
            chunks = hfs.stream_via_api(dataset_id=ds_id,config=d.get("config","default"),
                                        split=d.get("split","train"),n_samples=n,
                                        hf_token=state.get("hf_token"),progress_callback=cb)
            state["chunks"]+=chunks
            state["progress"]={"step":"done","current":n,"total":n,"pairs":len(chunks)}
        except Exception as e:
            state["progress"]={"step":f"error: {str(e)[:80]}","current":0,"total":1,"pairs":0}
    threading.Thread(target=run,daemon=True).start()
    return jsonify({"success":True})

@app.route("/api/hf_manual", methods=["POST"])
def hf_manual():
    d = request.json or {}
    ds_id = d.get("dataset_id","").strip(); n = int(d.get("n_samples",200000))
    if not ds_id: return jsonify({"error":"No dataset"}),400
    state["progress"] = {"step":f"Streaming {ds_id}...","current":0,"total":n,"pairs":0}
    def run():
        try:
            tok = state.get("hf_token")
            def cb(f,t,ds): state["progress"]={"step":f"Streamed {f:,}/{t:,}","current":f,"total":t,"pairs":0}
            chunks = hfs.stream(dataset_id=ds_id,config=d.get("config"),split=d.get("split","train"),
                                fields={"text":"text"},n_samples=n,hf_token=tok,progress_callback=cb)
            if not chunks:
                for fld in [{"instruction":"instruction","output":"output"},{"text":"content"},{"text":"code"}]:
                    chunks = hfs.stream(dataset_id=ds_id,config=d.get("config"),split=d.get("split","train"),
                                       fields=fld,n_samples=n,hf_token=tok,progress_callback=cb)
                    if chunks: break
            state["chunks"]+=chunks
            state["progress"]={"step":"done","current":n,"total":n,"pairs":len(chunks)}
        except Exception as e:
            state["progress"]={"step":f"error: {str(e)[:80]}","current":0,"total":1,"pairs":0}
    threading.Thread(target=run,daemon=True).start()
    return jsonify({"success":True})


@app.route("/api/hf_validate")
def hf_validate():
    """Validate a dataset before streaming — catches script errors early."""
    ds_id  = request.args.get("dataset_id","").strip()
    config = request.args.get("config","") or None
    if not ds_id: return jsonify({"error":"No dataset ID"}), 400
    result = hfs.validate_dataset(ds_id, config=config, token=state.get("hf_token"))
    return jsonify(result)

@app.route("/api/hf_dataset_info")
def hf_dataset_info():
    ds = request.args.get("dataset_id","").strip()
    if not ds: return jsonify({"error":"No ID"}),400
    return jsonify(hfs.get_dataset_info(ds,state.get("hf_token")))

# ── WEB COLLECTION ─────────────────────────────────────────────────────────────
@app.route("/api/web_sources")
def web_sources():
    return jsonify(webcol.get_available_sources())

@app.route("/api/web_collect", methods=["POST"])
def web_collect():
    d = request.json or {}
    topic = d.get("topic","").strip(); n = int(d.get("n_samples",10000))
    sources = d.get("sources",["web","wikipedia"])
    if not topic: return jsonify({"error":"Enter a topic"}),400
    state["progress"] = {"step":f"Collecting: {topic}...","current":0,"total":n,"pairs":0}
    def run():
        def cb(i,tot,src,col): state["progress"]={"step":f"From {src} ({i+1}/{tot})","current":col,"total":n,"pairs":0}
        chunks = webcol.collect(topic=topic,sources=sources,target_samples=n,progress_callback=cb)
        state["chunks"]+=chunks; state["web_topics"].append(topic)
        state["progress"]={"step":"done","current":n,"total":n,"pairs":len(chunks)}
    threading.Thread(target=run,daemon=True).start()
    return jsonify({"success":True,"estimated_time":webcol.estimate_time(sources,n)})

@app.route("/api/sample_guide")
def sample_guide():
    return jsonify(get_sample_guide())

# ── MODEL RESOLVER ─────────────────────────────────────────────────────────────
@app.route("/api/model_search")
def model_search():
    q = request.args.get("q","").strip()
    if not q: return jsonify([])
    return jsonify(resolver.search(q,hf_token=state.get("hf_token"),limit=20))

@app.route("/api/model_resolve", methods=["POST"])
def model_resolve():
    model_id = (request.json or {}).get("model_id","").strip()
    if not model_id: return jsonify({"error":"No model ID"}),400
    config = resolver.resolve(model_id,hf_token=state.get("hf_token"))
    state["sel_model"] = config
    return jsonify(config)

@app.route("/api/model_popular")
def model_popular():
    cat = request.args.get("category","all")
    return jsonify(resolver.get_popular(cat))

# ── DATA OVERVIEW ──────────────────────────────────────────────────────────────
@app.route("/api/data_overview")
def data_overview():
    c = state["chunks"]
    return jsonify({"total":len(c),"words":loader.word_count(c),"src":loader.get_source_stats(c)})

# ── CLEAN ──────────────────────────────────────────────────────────────────────
@app.route("/api/clean", methods=["POST"])
def clean():
    if not state["chunks"]: return jsonify({"error":"Add data first"}),400
    state["progress"]={"step":"Cleaning...","current":0,"total":1,"pairs":0}
    def run():
        def cb(c,t): state["progress"]={"step":f"Cleaning {c}/{t}","current":c,"total":t,"pairs":0}
        cleaned,stats = cleaner.clean_chunks(state["chunks"],progress_callback=cb)
        quality = cleaner.get_quality_score(cleaned)
        state["clean_chunks"],state["clean_stats"],state["quality"]=cleaned,stats,quality
        state["progress"]={"step":"done","current":1,"total":1,"pairs":0}
    threading.Thread(target=run,daemon=True).start()
    return jsonify({"success":True})

@app.route("/api/clean_stats")
def clean_stats():
    return jsonify({"stats":state["clean_stats"],"quality":state["quality"],"ready":len(state["clean_chunks"])>0})

# ── CATEGORIES / MATCH ─────────────────────────────────────────────────────────
@app.route("/api/categories")
def categories():
    return jsonify(matcher.get_all_categories())

@app.route("/api/match_model", methods=["POST"])
def match_model():
    cat = (request.json or {}).get("category","general")
    if not state["hw"]:
        info=hw.scan(); info["rec"]=hw.recommend_size(info); state["hw"]=info
    hw_info = {"ram_gb":state["hw"]["ram"]["gb"],"gpu":state["hw"]["gpu"]}
    result  = matcher.match(hw_info,cat)
    if result.get("best"):
        state["sel_model"]=result["best"]; state["sel_cat"]=cat
    return jsonify(result)

# ── PAIRS ──────────────────────────────────────────────────────────────────────
@app.route("/api/generate_pairs", methods=["POST"])
def generate_pairs():
    d = request.json or {}
    mode = d.get("mode","factual"); max_p = int(d.get("max_pairs",2000))
    chunks = state["clean_chunks"] if state["clean_chunks"] else state["chunks"]
    if not chunks: return jsonify({"error":"Add data first"}),400
    state["progress"]={"step":"Starting...","current":0,"total":1,"pairs":0}
    def run():
        def cb(c,t,n): state["progress"]={"step":f"Chunk {c}/{t}","current":c,"total":t,"pairs":n}
        pairs = genpairs.generate(chunks,mode=mode,pairs_per_chunk=4,max_pairs=max_p,progress_callback=cb)
        state["pairs"],state["sel_mode"]=pairs,mode
        genpairs.save_jsonl(pairs,PAIRS_FILE)
        state["progress"]={"step":"done","current":1,"total":1,"pairs":len(pairs)}
    threading.Thread(target=run,daemon=True).start()
    return jsonify({"success":True})

@app.route("/api/progress")
def progress():
    return jsonify(state["progress"])

@app.route("/api/pairs_stats")
def pairs_stats():
    if not state["pairs"]: return jsonify({"error":"No pairs"}),404
    s = genpairs.get_stats(state["pairs"])
    p = [{"q":x["instruction"],"think":x.get("thinking","")[:120]+"...","a":x.get("answer","")[:150]+"...","type":x.get("type","")} for x in state["pairs"][:3]]
    return jsonify({**s,"previews":p,"mode":state["sel_mode"]})

# ── NOTEBOOK ───────────────────────────────────────────────────────────────────
@app.route("/api/generate_notebook", methods=["POST"])
def generate_notebook():
    if not state["pairs"]: return jsonify({"error":"Generate pairs first"}),400
    m = state.get("sel_model")
    if not m: return jsonify({"error":"Select a model first"}),400
    model_id = m.get("id") or m.get("hf_id","")
    if model_id and "lora_r" not in m:
        m = resolver.resolve(model_id,hf_token=state.get("hf_token"))
    try:
        pn = float(str(m.get("params","7B")).replace("B","").replace("b",""))
    except Exception:
        pn = 7.0
    size = "small" if pn<3 else ("medium" if pn<6 else "large")
    model_info = {
        "hf_id":          m.get("id") or m.get("hf_id",""),
        "name":           m.get("name","model"),
        "size":           size,
        "gguf_gb":        m.get("gguf_gb",4.5),
        "colab_min":      m.get("colab_min",60),
        "unsloth":        m.get("unsloth",True),
        "arch":           m.get("arch","default"),
        "target_modules": m.get("target_modules",["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]),
    }
    mode_name = MODES.get(state["sel_mode"],MODES["factual"])["name"]
    nb_path   = gen_notebook(model_info,mode_name,OUTPUT_DIR)
    state["nb_path"] = nb_path
    return jsonify({"success":True})

@app.route("/api/download_notebook")
def download_notebook():
    if not state["nb_path"] or not os.path.exists(state["nb_path"]):
        return jsonify({"error":"Generate first"}),404
    return send_file(state["nb_path"],as_attachment=True,download_name="build_my_ai.ipynb",mimetype="application/json")

@app.route("/api/download_pairs")
def download_pairs():
    if not os.path.exists(PAIRS_FILE): return jsonify({"error":"No pairs"}),404
    return send_file(PAIRS_FILE,as_attachment=True,download_name="training_pairs.jsonl",mimetype="application/json")

if __name__ == "__main__":
    print("\n PersonalForge v10 — localhost:5000\n")
    app.run(host="0.0.0.0",port=5000,debug=False)
