.PHONY: gpu-check gpu-smoke gpu-train gpu-watch gpu-runs

gpu-check:
	python3 scripts/gpu_dispatch.py gpu-check

gpu-smoke:
	python3 scripts/gpu_dispatch.py smoke

gpu-train:
	python3 scripts/gpu_dispatch.py train

gpu-watch:
	gh run watch --repo MakSoS1/gpu-dispatch

gpu-runs:
	gh run list --repo MakSoS1/gpu-dispatch --workflow ecup-gpu.yml --limit 10

