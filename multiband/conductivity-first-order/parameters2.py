import numpy as np

u_temp = 116.032
params_ = [
    {
        "Ne": [350],
        "tmin": [-30],
        "tmax": [100],
        # "tmax": [450],
        "Nt": 2000,
        "T": [4/u_temp],#,0.5,0.54,0.56],
        "wd":  [5],
        "s": np.array([1,-1]),
        "m": [ np.array([0.85, 1.38]) ],
        "ef": [ np.array([290, 70]) ],
        "g1": [0.000001,0.00001,0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.4,1.6,1.8,2,3,5,7.5,10],
        "g": [ np.array([10,10]), np.array([0.01,10]), np.array([10,0.01]), np.array([0.01,0.01]), np.array([0.000001,0.000001]) ],
        "pre_d0": np.array([0.3,0.7]),
        "v": [0,0.2],
        "A0": [1],
        "tau": [1.6],
        "w":  [1],
        "A0_pr": [0],
        "te": 0,
        "tau_pr": [1.5],
        "w_pr": [1],
        "t_delay": [0],
        "te_pr": 0
    },
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
                                                    for g1 in p["g1"]:
                                                        for g2 in p["g1"]:
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
                                                                                "g": np.array([g1,g2]),
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
