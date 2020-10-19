import numpy as np

u_temp = 116.032
x=1
params_ = [
    {
        "Ne": [500],
        "tmin": [-75],
        "tmax": [450],
        "Nt": 3000,
        "T": [1/u_temp],#,0.5,0.54,0.56],
        "wd":  [1],
        "s": np.array([1]),
        "m": [ np.array([0.78]) ],
        "ef": [ np.array([100]) ],
        "g": [ np.array([5])*0.1258396834202397*2 ],
        "U": [ np.array([[0.46]]) ],
        "A0": [-25],
        "tau": [5.37/x],
        "w":  [0.04*x],
        "A0_pr": [0.1,0],
        "te": -40/x,
        "tau_pr": [60],
        "w_pr": [0.3, 0.27],
        "t_delay": np.linspace(80,200,80),
        "te_pr": 0
    },
    {
        "Ne": [500],
        "tmin": [-75],
        "tmax": [450],
        "Nt": 3000,
        "T": [1/u_temp],#,0.5,0.54,0.56],
        "wd":  [1],
        "s": np.array([1]),
        "m": [ np.array([0.78]) ],
        "ef": [ np.array([100]) ],
        "g": [ np.array([5])*0.1258396834202397*2 ],
        "U": [ np.array([[0.46]]) ],
        "A0": [0],
        "tau": [5.37/x],
        "w":  [0.04*x],
        "A0_pr": [0.1],
        "te": -40/x,
        "tau_pr": [60],
        "w_pr": [0.3, 0.27],
        "t_delay": np.linspace(80,200,80),
        "te_pr": 0
    }
]


params = []
for p in params_:
    for A0 in p["A0"]:
        for A0_pr in p["A0_pr"]:
            for Ne in p["Ne"]:
                for tmin in p["tmin"]:
                    for tmax in p["tmax"]:
                        for T in p["T"]:
                            for wd in p["wd"]:
                                for m in p["m"]:
                                    for ef in p["ef"]:
                                        for U in p["U"]:
                                            for tau in p["tau"]:
                                                for w in p["w"]:
                                                    for tau_pr in p["tau_pr"]:
                                                        for w_pr in p["w_pr"]:
                                                            for g in p["g"]:
                                                                for t_delay in p["t_delay"]:
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
                                                                        "U": U,
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
