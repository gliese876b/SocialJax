import sys, pickle, json
sys.path.append('/home/rlteam/Desktop/SocialJax')

import jax
import jax.numpy as jnp
import socialjax
from algorithms.IPPO.ippo_cnn_harvest_timeout import run_evaluation_episodes

def main(params_path, config_path, result_path):
    with open(params_path, 'rb') as f:
        params = pickle.load(f)
    with open(config_path, 'r') as f:
        config = json.load(f)

    eval_seed = config["_eval_seed"]
    num_episodes = config["EVAL_EPISODES"]

    results = run_evaluation_episodes(params, config, eval_seed, num_episodes)

    # Convert JAX arrays to lists for JSON
    summary = {k: [float(x) for x in v] if hasattr(v, '__iter__') else float(v)
               for k, v in results["summary"].items()}

    import numpy as np
    raw = {k: np.array(v).tolist() for k, v in results["raw"].items()}

    with open(result_path, 'w') as f:
        json.dump({"summary": summary, "raw": raw}, f)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])