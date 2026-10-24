"""Teste de carga simples para demonstrar escalabilidade (sem dependências extras).

Gera demanda em ondas (baixa -> pico -> baixa) contra /predict e reporta vazão e
latências. Útil para observar o HPA/Cloud Run criando réplicas e o Grafana reagindo.

    python scripts/load_test.py --url http://localhost:8000 --stages 10:30 50:60 10:30
    (estágio = concorrência:segundos)
"""

import argparse
import statistics
import sys
import threading
import time

import requests


def worker(url: str, payload: dict, stop: threading.Event, latencies: list, errors: list) -> None:
    session = requests.Session()
    while not stop.is_set():
        start = time.perf_counter()
        try:
            resp = session.post(f"{url}/predict", json=payload, timeout=30)
            ok = resp.status_code == 200
        except requests.RequestException:
            ok = False
        (latencies if ok else errors).append(time.perf_counter() - start)


def run_stage(url: str, payload: dict, concurrency: int, seconds: int) -> dict:
    stop = threading.Event()
    latencies: list[float] = []
    errors: list[float] = []
    threads = [threading.Thread(target=worker, args=(url, payload, stop, latencies, errors), daemon=True)
               for _ in range(concurrency)]
    for t in threads:
        t.start()
    time.sleep(seconds)
    stop.set()
    for t in threads:
        t.join(timeout=35)
    lat = sorted(latencies)
    pct = lambda q: lat[min(int(q * len(lat)), len(lat) - 1)] * 1000 if lat else float("nan")
    return {"concorrencia": concurrency, "segundos": seconds, "req_s": len(lat) / seconds, "erros": len(errors),
            "p50_ms": pct(0.5), "p95_ms": pct(0.95), "media_ms": statistics.mean(lat) * 1000 if lat else float("nan")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--stages", nargs="+", default=["5:20", "40:40", "5:20"])
    parser.add_argument("--batch", type=int, default=10, help="pacientes por requisição")
    args = parser.parse_args()

    examples = requests.get(f"{args.url}/examples", params={"n": args.batch}, timeout=30).json()
    payload = {"patients": [{"patient_id": e["patient_id"], "features": e["features"]} for e in examples]}

    print(f"{'conc.':>6} {'dur.':>5} {'req/s':>8} {'erros':>6} {'p50 ms':>8} {'p95 ms':>8}")
    for stage in args.stages:
        concurrency, seconds = map(int, stage.split(":"))
        r = run_stage(args.url, payload, concurrency, seconds)
        print(f"{r['concorrencia']:>6} {r['segundos']:>5} {r['req_s']:>8.1f} {r['erros']:>6} "
              f"{r['p50_ms']:>8.0f} {r['p95_ms']:>8.0f}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
