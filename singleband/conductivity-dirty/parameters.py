import numpy as np

u_temp = 116.032

params_ = [
    {
        "Ne": [600],
        "tmin": [-75],
        "tmax": [450],
        "Nt": 3000,
        "T": [4/u_temp],#,0.5,0.54,0.56],
        "wd":  [2],
        "s": np.array([1]),
        "m": [ np.array([0.78]) ],
        "ef": [ np.array([100]) ],
        "g": [ np.array([x]) for x in
              np.array([0.000001,0.00001,0.0001,0.001,0.01,0.05,0.1,0.15,0.2,0.3,0.4,0.45,0.5,0.55,0.6,0.7,0.8,0.9,0.95,1,1.05,1.1,1.2,1.3, 20])*0.1258396834202397*2 ],
        "U": [ np.array([[0.46]]) ],
        "A0": [-25],
        "tau": [5.37],
        "w":  [0.04],
        "A0_pr": [0],
        "te": -40,
        "tau_pr": [5.37],
        "w_pr": [0.04],
        "t_delay": [0],
        "te_pr": -40
    },
    {
        "Ne": [600],
        "tmin": [-75],
        "tmax": [450],
        "Nt": 3000,
        "T": [4/u_temp],#,0.5,0.54,0.56],
        "wd":  [2],
        "s": np.array([1]),
        "m": [ np.array([0.78]) ],
        "ef": [ np.array([100]) ],
        "g": [ np.array([20])*0.1258396834202397*2 ],
        "U": [ np.array([[0.46]]) ],
        "A0": [-25],
        "tau": [5.37],
        "w":  [0.04],
        "A0_pr": [-2.5],
        "te": -40,
        "tau_pr": [5.37],
        "w_pr": [0.04],
        "t_delay": np.linspace(50,200,128),
        "te_pr": -40
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
