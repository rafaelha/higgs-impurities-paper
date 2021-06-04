import numpy as np

u_temp = 116.032
params_ = [
    {
        "Ne": [800],
        "tmin": [-50],
        "tmax": [200],
        # "tmax": [450],
        "Nt": 3000,
        "T": [4/u_temp],#,0.5,0.54,0.56],
        "wd":  [5],
        "s": np.array([1,-1]),
        "m": [ np.array([0.85, 1.38]) ],
        "ef": [ np.array([290, 70]) ],
        "g": [ np.array([10,5])],
        "pre_d0": np.array([0.3,0.7]),
        "v": [0.2],
        "A0": [1],
        "tau": [8],
        "w":  [0.7],
        "A0_pr": [0.1],
        "te": 0,
        "tau_pr": [1.5],
        "w_pr": [1],
        "t_delay": np.linspace(0,100,100),
        "te_pr": 0
    },
    {
        "Ne": [800],
        "tmin": [-50],
        "tmax": [200],
        # "tmax": [450],
        "Nt": 3000,
        "T": [4/u_temp],#,0.5,0.54,0.56],
        "wd":  [5],
        "s": np.array([1,-1]),
        "m": [ np.array([0.85, 1.38]) ],
        "ef": [ np.array([290, 70]) ],
        "g": [ np.array([10,5])],
        "pre_d0": np.array([0.3,0.7]),
        "v": [0.2],
        "A0": [1],
        "tau": [8],
        "w":  [0.3],
        "A0_pr": [0.1],
        "te": 0,
        "tau_pr": [1.5],
        "w_pr": [1],
        "t_delay": np.linspace(0,100,100),
        "te_pr": 0
    },
    {
        "Ne": [800],
        "tmin": [-50],
        "tmax": [200],
        # "tmax": [450],
        "Nt": 3000,
        "T": [4/u_temp],#,0.5,0.54,0.56],
        "wd":  [5],
        "s": np.array([1,-1]),
        "m": [ np.array([0.85, 1.38]) ],
        "ef": [ np.array([290, 70]) ],
        "g": [ np.array([10,5])],
        "pre_d0": np.array([0.3,0.7]),
        "v": [0.2],
        "A0": [1],
        "tau": [2],
        "w":  [0.5],
        "A0_pr": [0.1],
        "te": 0,
        "tau_pr": [1.5],
        "w_pr": [1],
        "t_delay": np.linspace(0,100,100),
        "te_pr": 0
    },
    {
        "Ne": [800],
        "tmin": [-50],
        "tmax": [200],
        # "tmax": [450],
        "Nt": 3000,
        "T": [4/u_temp],#,0.5,0.54,0.56],
        "wd":  [5],
        "s": np.array([1,-1]),
        "m": [ np.array([0.85, 1.38]) ],
        "ef": [ np.array([290, 70]) ],
        "g": [ np.array([10,5])],
        "pre_d0": np.array([0.3,0.7]),
        "v": [0.2],
        "A0": [1],
        "tau": [8],
        "w":  [0.7],
        "A0_pr": [0],
        "te": 0,
        "tau_pr": [1.5],
        "w_pr": [1],
        "t_delay": [0],
        "te_pr": 0
    },
    {
        "Ne": [800],
        "tmin": [-50],
        "tmax": [200],
        # "tmax": [450],
        "Nt": 3000,
        "T": [4/u_temp],#,0.5,0.54,0.56],
        "wd":  [5],
        "s": np.array([1,-1]),
        "m": [ np.array([0.85, 1.38]) ],
        "ef": [ np.array([290, 70]) ],
        "g": [ np.array([10,5])],
        "pre_d0": np.array([0.3,0.7]),
        "v": [0.2],
        "A0": [1],
        "tau": [8],
        "w":  [0.3],
        "A0_pr": [0],
        "te": 0,
        "tau_pr": [1.5],
        "w_pr": [1],
        "t_delay": [0],
        "te_pr": 0
    },
    {
        "Ne": [800],
        "tmin": [-50],
        "tmax": [200],
        # "tmax": [450],
        "Nt": 3000,
        "T": [4/u_temp],#,0.5,0.54,0.56],
        "wd":  [5],
        "s": np.array([1,-1]),
        "m": [ np.array([0.85, 1.38]) ],
        "ef": [ np.array([290, 70]) ],
        "g": [ np.array([10,5])],
        "pre_d0": np.array([0.3,0.7]),
        "v": [0.2],
        "A0": [1],
        "tau": [2],
        "w":  [0.5],
        "A0_pr": [0],
        "te": 0,
        "tau_pr": [1.5],
        "w_pr": [1],
        "t_delay": [0],
        "te_pr": 0
    }
]



params = []
for p in params_:
    for Ne in p["Ne"]:
        for tmin in p["tmin"]:
            for tmax in p["tmax"]:
                for T in p["T"]:
                    for wd in p["wd"]:
                        for m in p["m"]:
                            for ef in p["ef"]:
                                for v in p["v"]:
                                    for tau in p["tau"]:
                                        for w in p["w"]:
                                            for tau_pr in p["tau_pr"]:
                                                for w_pr in p["w_pr"]:
                                                    for g in p["g"]:
                                                        for A0 in p["A0"]:
                                                            for A0_pr in p["A0_pr"]:
                                                                for t_delay in p["t_delay"]:
                                                                    if A0_pr != 0 or A0 != 0:
                                                                        params.append({
                                                                            "Ne": Ne,
                                                                            "tmin": tmin,
                                                                            "tmax": tmax,
                                                                            "Nt": p["Nt"],
                                                                            "T": T,
                                                                            "wd": wd,
                                                                            "s": p["s"],
                                                                            "m": m,
                                                                            "ef": ef,
                                                                            "g": g,
                                                                            "pre_d0": p["pre_d0"],
                                                                            "v": v,
                                                                            "A0": A0,
                                                                            "tau": tau,
                                                                            "w":  w,
                                                                            "te": p["te"],
                                                                            "A0_pr": A0_pr,
                                                                            "tau_pr": tau_pr,
                                                                            "w_pr": w_pr,
                                                                            "t_delay": t_delay,
                                                                            "te_pr": p["te_pr"]
                                                                        })
print('parameters generated')
print(len(params))
