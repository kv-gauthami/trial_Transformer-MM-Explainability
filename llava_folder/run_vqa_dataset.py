"""
Layer-wise A / R maps for a closed-answer spatial VQA dataset (JSONL), LLaVA-1.5.
Default options and prompt parsing fit What'sUp / AdaptVis Controlled_Images_B
(left / right / front / behind).

Input record (one per line):
  {"id": 0,
   "question": "<image>\\nUSER: Where is the mug in relation to the knife? Answer with left, right, front or behind.\\nASSISTANT:",
   "answer": ["Right"]}

Images (the JSONL has no image path), one of:
  --image_json  the dataset's own JSON (list, or dict holding a list) whose i-th entry has
                "image_path" (What'sUp / AdaptVis controlled_clevr_dataset.json). Record id i -> entry i.
  --image_dir + --image_pattern "{id}.jpeg"
  an "image" field inside each record.
Always run once with --check_images first: it prints the id -> image mapping and loads no model.

Per sample, ONE forward pass:
  1. q = last prompt token (after "ASSISTANT:"), the position that predicts the answer
     (plays the role of [CLS] in Chefer et al.).
  2. The options are scored by log-prob at q; argmax = model's answer
     (= greedy decoding restricted to the allowed options). Nothing is generated.
  3. One placeholder token is appended at t = q + 1 so explain_word() reads row t-1 = q.
     The causal mask keeps row q, its logits and its maps unaffected by that token.
  4. Maps for the PREDICTED option, and for the GROUND-TRUTH option when the model is wrong.

Outputs in --out_dir:
  <save_name>.pkl            {(id, "pred:Right" | "gt:Left", map_type, layer): array[24, 24]}
  <save_name>_results.jsonl  per sample: objects, gt, pred, correct, option probs, image, ...
  <save_name>_meta.json      run settings
  plots/                     first --plot_first samples only

Example:
  python llava_folder/run_vqa_dataset.py \\
      --data Controlled_Dataset_Adaptvis/prompts/Controlled_Images_B_with_answer_four_options.jsonl \\
      --image_json Controlled_Dataset_Adaptvis/controlled_clevr \\
      --out_dir llava_folder/controlled_B --plot_first 5 --check_images
"""
import argparse
import json
import os
import pickle
import re
import sys
import time

import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

HERE = os.path.dirname(os.path.abspath(__file__))
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
DEFAULT_OPTIONS = ["Left", "Right", "Front", "Behind"]
IMAGE_KEYS = ("image", "image_path", "image_file", "img", "file_name")
# "Where is the mug in relation to the knife?" / "Where are the sunglasses in relation to the tape?"
RELATION_RE = re.compile(r"where (?:is|are) the (.+?) in relation to the (.+?)\?", re.IGNORECASE)
# "Where is the purse in the photo?"
SINGLE_RE = re.compile(r"where (?:is|are) the (.+?) in the (?:photo|image|picture)", re.IGNORECASE)
PLOTS = [("A", "per_layer"), ("A", "global"), ("R_layer", "global"), ("R_cum", "per_layer")]


# ----------------------------------------------------------------------------- args
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="JSONL file")
    p.add_argument("--model_id", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--image_json", default=None,
                   help="dataset JSON whose i-th entry has 'image_path' (record id i -> entry i)")
    p.add_argument("--image_root", default=None,
                   help="folder that relative paths from --image_json are joined to")
    p.add_argument("--image_dir", default=None, help="folder for --image_pattern / record image fields")
    p.add_argument("--image_pattern", default=None, help='e.g. "{id}.jpeg"')
    p.add_argument("--check_images", action="store_true",
                   help="print the id -> image mapping, check files exist, and exit (no model)")
    p.add_argument("--options", nargs="+", default=DEFAULT_OPTIONS)
    p.add_argument("--no_case_variants", action="store_true",
                   help="score only the exact option spelling (default also scores lowercase)")
    p.add_argument("--start", type=int, default=0, help="skip the first N records")
    p.add_argument("--limit", type=int, default=None, help="process at most N records")
    p.add_argument("--out_dir", default=os.path.join(HERE, "vqa_maps"))
    p.add_argument("--save_name", default="relevancy_maps")
    p.add_argument("--save_every", type=int, default=25, help="checkpoint the pkl every N samples")
    p.add_argument("--resume", action="store_true", help="skip ids already in the results file")
    p.add_argument("--plot_first", type=int, default=0, help="make plots for the first N samples")
    p.add_argument("--dtype", default="float32", choices=list(DTYPES))
    p.add_argument("--device", default="cuda")
    return p.parse_args()


# ----------------------------------------------------------------------------- data helpers
def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_image_list(path):
    with open(path) as f:
        obj = json.load(f)
    if isinstance(obj, dict):                 # e.g. {"data": [...]} or {"annotations": [...]}
        lists = [v for v in obj.values() if isinstance(v, list)]
        if len(lists) != 1:
            raise ValueError(f"{path}: expected a list or a dict holding one list.")
        obj = lists[0]
    paths = []
    for i, e in enumerate(obj):
        if isinstance(e, str):
            paths.append(e)
            continue
        key = next((k for k in IMAGE_KEYS if k in e), None)
        if key is None:
            raise ValueError(f"{path}: entry {i} has none of {IMAGE_KEYS}: {list(e)}")
        paths.append(e[key])
    return paths


def resolve_image(rec, args, image_list):
    if image_list is not None:
        i = int(rec["id"])
        if i >= len(image_list):
            raise IndexError(f"id {i} but --image_json has only {len(image_list)} entries.")
        path = image_list[i]
        root = args.image_root
    else:
        key = next((k for k in IMAGE_KEYS if rec.get(k)), None)
        if key is not None:
            path = rec[key]
        elif args.image_pattern is not None:
            path = args.image_pattern.format(**rec)
        else:
            raise ValueError(f"Record {rec.get('id')}: no image. Pass --image_json, "
                             "or --image_dir + --image_pattern.")
        root = args.image_dir
    if root and not os.path.isabs(path):
        path = os.path.join(root, path)
    return path


def extract_objects(question):
    m = RELATION_RE.search(question)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m = SINGLE_RE.search(question)
    if m:
        return m.group(1).strip(), None
    return None, None


def normalize_answer(ans, options):
    """['Right'] / 'right.' / ' RIGHT ' -> 'Right' (or None if not an option)."""
    if isinstance(ans, (list, tuple)):
        ans = ans[0] if ans else None
    if ans is None:
        return None
    a = re.sub(r"[^a-z]", "", str(ans).lower())
    return next((o for o in options if o.lower() == a), None)


# ----------------------------------------------------------------------------- model helpers
def option_token_ids(tokenizer, prompt, options, case_variants=True):
    """First token of each option as it would follow the prompt -> {option: [ids]}."""
    base = tokenizer(prompt).input_ids
    out, notes = {}, []
    for opt in options:
        variants = {opt, opt.lower(), opt.capitalize()} if case_variants else {opt}
        ids = set()
        for v in sorted(variants):
            full = tokenizer(prompt + " " + v).input_ids
            if full[:len(base)] != base:
                raise ValueError(f"Tokenization of the prompt changed when appending {v!r}.")
            new = full[len(base):]
            if new and tokenizer.convert_ids_to_tokens(new[0]) == "▁":   # bare-space piece
                new = new[1:]
            if not new:
                raise ValueError(f"Option {v!r} produced no new tokens.")
            if len(new) > 1:
                notes.append(f"{v!r} -> {tokenizer.convert_ids_to_tokens(new)} (first token used)")
            ids.add(new[0])
        out[opt] = sorted(ids)
    firsts = [i for ids in out.values() for i in ids]
    if len(firsts) != len(set(firsts)):
        raise ValueError(f"Two options share a first token: "
                         f"{ {o: tokenizer.convert_ids_to_tokens(i) for o, i in out.items()} }")
    return out, notes


def save_checkpoint(records, meta, args):
    os.makedirs(args.out_dir, exist_ok=True)
    pkl = os.path.join(args.out_dir, f"{args.save_name}.pkl")
    with open(pkl + ".tmp", "wb") as f:
        pickle.dump(records, f)
    os.replace(pkl + ".tmp", pkl)             # never leaves a half-written pkl
    with open(os.path.join(args.out_dir, f"{args.save_name}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def make_plots(viz, image_rgb, maps, wdir):
    arrays = {"A": maps.A, "R_layer": maps.R_layer, "R_cum": maps.R_cum}
    for mtype, sc in PLOTS:
        viz.plot_layer_grid(image_rgb, arrays[mtype], f"{mtype} '{maps.word}'",
                            os.path.join(wdir, f"{mtype}_{sc}.png"), scale=sc)
    viz.plot_summary(image_rgb, maps, os.path.join(wdir, "summary.png"))
    viz.plot_final(image_rgb, maps, os.path.join(wdir, "R_final.png"))


# ----------------------------------------------------------------------------- main
def main():
    args = parse_args()
    data = load_jsonl(args.data)[args.start:]
    if args.limit is not None:
        data = data[:args.limit]
    image_list = load_image_list(args.image_json) if args.image_json else None

    if args.check_images:
        missing = 0
        for k, rec in enumerate(data):
            path = resolve_image(rec, args, image_list)
            ok = os.path.exists(path)
            missing += not ok
            if k < 8 or not ok:
                o1, o2 = extract_objects(rec["question"])
                print(f"[{rec['id']}] {'ok  ' if ok else 'MISSING'} {path}   "
                      f"q: {o1} vs {o2}   gt: {normalize_answer(rec.get('answer'), args.options)}")
        print(f"\n{len(data) - missing}/{len(data)} images found.")
        print("Check that the file names match the questions above (e.g. 'mug ... knife' -> "
              "an image of a mug and a knife) before running the model.")
        return

    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from llava_explainability.pipeline import prepare_model_for_relevance
    from llava_explainability.hook_handler import AttentionHookManager
    from llava_explainability.explainer import LlavaLayerwiseExplainer
    from llava_explainability.token_utils import image_token_span
    from llava_explainability.storage import maps_to_records
    from llava_explainability import visualize as viz

    dtype = DTYPES[args.dtype]
    os.makedirs(args.out_dir, exist_ok=True)
    results_path = os.path.join(args.out_dir, f"{args.save_name}_results.jsonl")
    pkl_path = os.path.join(args.out_dir, f"{args.save_name}.pkl")

    records, done = {}, set()
    if args.resume and os.path.exists(results_path):
        with open(results_path) as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                records = pickle.load(f)
        print(f"Resuming: {len(done)} samples already done.")
    elif os.path.exists(results_path):
        os.remove(results_path)               # fresh run: start the results file over

    torch.manual_seed(0)
    processor = AutoProcessor.from_pretrained(args.model_id)
    tokenizer = processor.tokenizer
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=dtype, device_map=args.device, attn_implementation="eager")
    prepare_model_for_relevance(model)
    image_token_id = model.config.image_token_index

    meta = {"model_id": args.model_id, "data": os.path.abspath(args.data),
            "image_json": args.image_json, "image_root": args.image_root,
            "options": args.options, "case_variants": not args.no_case_variants,
            "dtype": args.dtype, "row_convention": "t-1 (last prompt token)",
            "key_format": "(id, '<pred|gt>:<option>', map_type, layer)"}

    hook_mgr = AttentionHookManager(model)
    explainer = LlavaLayerwiseExplainer(model, hook_mgr)
    n_done = n_correct = n_since_save = 0
    t0 = time.time()
    try:
        for rec in data:
            sid = rec["id"]
            if sid in done:
                continue
            prompt = rec["question"].replace("\\n", "\n")
            obj1, obj2 = extract_objects(prompt)
            gt = normalize_answer(rec.get("answer"), args.options)
            img_path = resolve_image(rec, args, image_list)
            if not os.path.exists(img_path):
                print(f"[{sid}] image not found: {img_path}  (skipped)")
                continue
            image = Image.open(img_path).convert("RGB")

            inputs = processor(text=prompt, images=image, return_tensors="pt").to(args.device)
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
            opt_ids, notes = option_token_ids(tokenizer, prompt, args.options,
                                              case_variants=not args.no_case_variants)
            if notes and n_done == 0:
                print("Tokenizer note:", "; ".join(notes))

            # append one placeholder so t = q + 1 and explain_word reads row q
            ids, am = inputs["input_ids"], inputs["attention_mask"]
            q = ids.shape[1] - 1
            t = q + 1
            inputs["input_ids"] = torch.cat([ids, ids.new_tensor([[opt_ids[args.options[0]][0]]])], dim=1)
            inputs["attention_mask"] = torch.cat([am, am.new_ones((1, 1))], dim=1)
            img_start, n_img, grid = image_token_span(inputs["input_ids"][0].tolist(), image_token_id)

            explainer.forward(inputs)
            logp = torch.log_softmax(explainer.outputs.logits[0, q].float(), dim=-1)
            scores = {o: torch.logsumexp(logp[i], dim=0).item() for o, i in opt_ids.items()}
            pred = max(scores, key=scores.get)
            probs = dict(zip(scores, torch.softmax(torch.tensor(list(scores.values())), 0).tolist()))
            top1 = tokenizer.convert_ids_to_tokens(int(logp.argmax()))

            targets = [("pred", pred)] + ([("gt", gt)] if gt is not None and gt != pred else [])
            row = {"id": sid, "object": obj1, "reference": obj2, "image": img_path,
                   "gt": gt, "pred": pred, "correct": (gt == pred) if gt is not None else None,
                   "option_prob": probs, "option_logprob": scores,
                   "vocab_top1_token": top1, "query_pos": q, "maps": {}}

            for role, opt in targets:
                tid = max(opt_ids[opt], key=lambda i: logp[i].item())    # best-scoring spelling
                ids_t = inputs["input_ids"].clone()
                ids_t[0, t] = tid
                word = f"{role}:{opt}"
                maps = explainer.explain_word(ids_t, word, t, img_start, n_img, grid)
                records.update(maps_to_records(sid, maps))
                row["maps"][word] = {"token": tokenizer.convert_ids_to_tokens(tid),
                                     "vocab_prob": maps.target_prob, "logit": maps.target_logit}
                if n_done < args.plot_first:
                    make_plots(viz, viz.pixel_values_to_image(inputs["pixel_values"]), maps,
                               os.path.join(args.out_dir, "plots", f"id{sid}_{role}_{opt}"))

            with open(results_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            n_done += 1
            n_correct += int(bool(row["correct"]))
            n_since_save += 1
            mark = "ok " if row["correct"] else ("ERR" if row["correct"] is False else " ? ")
            print(f"[{sid}] {mark} {obj1} vs {obj2}: gt={gt} pred={pred} "
                  f"p={probs[pred]:.2f} top1={top1!r}  acc={n_correct / n_done:.3f}")

            explainer.outputs = None              # free the graph before the next sample
            hook_mgr.clear()
            del inputs, ids_t, logp
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
            if n_since_save >= args.save_every:
                save_checkpoint(records, meta, args)
                n_since_save = 0
    finally:
        hook_mgr.remove()
        save_checkpoint(records, meta, args)

    dt = time.time() - t0
    if n_done:
        print(f"\nDone: {n_done} samples in {dt / 60:.1f} min ({dt / n_done:.1f} s/sample). "
              f"Accuracy: {n_correct / n_done:.3f}")
    print(f"Maps:    {pkl_path}\nResults: {results_path}")


if __name__ == "__main__":
    main()