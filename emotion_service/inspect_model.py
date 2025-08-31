# inspect_model_loader.py
"""
Load a checkpoint using your predict.py loader and introspect its forward behavior.
Run from the same directory where you ran predict.py (your current working dir).
Example:
  python inspect_model_loader.py --model saved_models/multimodal_best.pth --device cpu
"""
import argparse
import torch
import traceback

def import_predict_module():
    # try several import paths depending on where you run this script
    candidates = ["inference.predict", "emotion_service.inference.predict", "predict", "emotion_service.inference.predict"]
    last_exc = None
    for name in candidates:
        try:
            mod = __import__(name, fromlist=["*"])
            print(f"Imported module: {name}")
            return mod
        except Exception as e:
            last_exc = e
    print("Failed to import predict module. Last error:")
    raise last_exc

def run_inspect(model_path, device_str="cpu"):
    mod = import_predict_module()
    # Expect load_multimodal_model to be present
    if not hasattr(mod, "load_multimodal_model"):
        raise RuntimeError("predict.py does not expose load_multimodal_model. Confirm file/inference layout.")
    device = torch.device(device_str)
    print(f"Loading model checkpoint via load_multimodal_model('{model_path}', device={device}) ...")
    model, meta = mod.load_multimodal_model(model_path, device)
    print("Model loaded. meta keys:", list(meta.keys()) if isinstance(meta, dict) else meta)
    print("\n=== model repr ===")
    try:
        print(model)
    except Exception as e:
        print("Could not print model repr:", e)

    print("\n=== named_children (top-level) ===")
    try:
        for name, m in list(model.named_children())[:100]:
            print("-", name, "->", type(m))
    except Exception as e:
        print("named_children failed:", e)

    print("\n=== named_parameters (sample) ===")
    try:
        for i, (n, p) in enumerate(model.named_parameters()):
            print(f"param {i}: {n} shape={tuple(p.shape)}")
            if i >= 60:
                break
    except Exception as e:
        print("named_parameters failed:", e)

    # Build realistic dummy tensors (match your preprocess shapes)
    img = torch.randn(1, 3, 224, 224, device=device)
    aud = torch.randn(1, 1, 128, 250, device=device)  # time dim ~250 (arbitrary)

    def safe_call(*args, **kwargs):
        try:
            with torch.no_grad():
                return model(*args, **kwargs)
        except Exception as e:
            return e

    print("\n=== Try model(img, aud) ===")
    out = safe_call(img, aud)
    print("type:", type(out))
    if isinstance(out, tuple):
        print("tuple length:", len(out))
        for i, x in enumerate(out):
            try:
                print(f"  [{i}] type={type(x)} shape={getattr(x,'shape',None)}")
            except Exception as e:
                print(f"  [{i}] repr error: {e}")
    elif isinstance(out, dict):
        print("dict keys:", list(out.keys()))
        for k, v in out.items():
            try:
                print(f"  key={k} type={type(v)} shape={getattr(v,'shape',None)}")
            except Exception as e:
                print(f"  key={k} repr error: {e}")
    else:
        print("output repr:", repr(out)[:1000])

    print("\n=== Search for candidate embedding methods/attrs ===")
    candidates = ["get_embeddings", "get_features", "encode", "encode_image", "encode_audio",
                  "embed", "embedder", "projection", "projection_head", "encoder", "backbone",
                  "feature_extractor", "fusion", "fusion_net", "classifier", "head", "fc", "forward_features"]
    for c in candidates:
        if hasattr(model, c):
            obj = getattr(model, c)
            print("HAS:", c, "->", type(obj))

    print("\n=== named_modules that are Linear (in/out features) ===")
    import torch.nn as nn
    try:
        for name, m in model.named_modules():
            if isinstance(m, nn.Linear):
                print(f"Linear {name} -> in={m.in_features} out={m.out_features}")
    except Exception as e:
        print("named_modules scan failed:", e)

    print("\n=== Done ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="path to .pth checkpoint")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    try:
        run_inspect(args.model, args.device)
    except Exception:
        traceback.print_exc()
