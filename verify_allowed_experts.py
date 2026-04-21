import argparse
import runpy
import sys
import torch

def patch_ffnexperts():
    from src.models.addons.moe import FFNExperts

    orig_pre_forward = FFNExperts.pre_forward

    def wrapped_pre_forward(self, hidden_states, *args, **kwargs):
        allowed = None
        if hasattr(self, "_parse_allowed_experts"):
            allowed = self._parse_allowed_experts()

        if allowed is None:
            return orig_pre_forward(self, hidden_states, *args, **kwargs)

        rw = self.global_hidden_dict[self.read_routing_weights_key]

        if hasattr(self, "_apply_allowed_experts_mask"):
            rw = self._apply_allowed_experts_mask(rw)
            self.global_hidden_dict[self.read_routing_weights_key] = rw

        allowed_set = set(int(x) for x in allowed)
        num_experts = rw.shape[-1]

        if not getattr(self, "_verify_printed_once", False):
            print("[VERIFY] FFNExperts module key:", self.read_routing_weights_key)
            print("[VERIFY] num_experts:", num_experts, " topk_value:", self.topk_value)
            print("[VERIFY] allowed size:", len(allowed_set))
            self._verify_printed_once = True

        with torch.no_grad():
            if self.topk_value is not None:
                k = min(int(self.topk_value), num_experts)
                _, idx = torch.topk(rw, k, dim=-1)
                idx_flat = idx.reshape(-1).tolist()
                bad = [int(i) for i in idx_flat if int(i) not in allowed_set]
                if bad:
                    uniq = sorted(set(bad))[:20]
                    raise AssertionError(
                        f"[VERIFY FAIL] Found topk indices outside allowed AFTER mask! "
                        f"Example bad experts={uniq} (show up to 20)."
                    )
            else:
                mask = torch.zeros(num_experts, device=rw.device, dtype=rw.dtype)
                mask[list(allowed_set)] = 1
                outside_mass = (rw * (1 - mask)).abs().max().item()
                if outside_mass > 1e-6:
                    raise AssertionError(
                        f"[VERIFY FAIL] Found nonzero routing mass outside allowed AFTER mask: max={outside_mass}"
                    )

        return orig_pre_forward(self, hidden_states, *args, **kwargs)

    FFNExperts.pre_forward = wrapped_pre_forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--launch_args",
        nargs=argparse.REMAINDER,
        help="Args passed to src/launch_single_process.py after '--launch_args'",
    )
    args = ap.parse_args()

    if not args.launch_args:
        print(
            "Usage example:\n"
            "python verify_allowed_experts.py --launch_args "
            "--gin_files <...> --gin_bindings '<...>'\n"
        )
        sys.exit(2)

    patch_ffnexperts()
    sys.argv = ["src/launch_single_process.py"] + args.launch_args
    runpy.run_path("src/launch_single_process.py", run_name="__main__")


if __name__ == "__main__":
    main()