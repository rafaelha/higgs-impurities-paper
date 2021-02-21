import numpy as np

u_temp = 116.032
u_t = 6.58285E-2
u_e = 10
u_conductivity = 881.553 #Ohm^-1 cm-1
meV_to_THz = 0.2417990504024
u_w = u_e*meV_to_THz

params_ = [
    {
        # "Ne": [500],
        # "eta":  [0.025],
        # # "w":  np.array([0.3,0.4,0.5,0.6])/u_w,
        # "w":  np.linspace(0.1,0.85,24),
        # "T": np.array([0.4])/u_temp,
        # "wd":  [5],
        # "s": np.array([1,-1]),
        # "m": [ np.array([0.85, 1.38]) ],
        # "ef": [ np.array([290, 70]) ],
        # "g": [ np.array([0.001,0.001])],
        # "pre_d0": np.array([0.3,0.7]),
        # "v": [0.02]
        "Ne": [500],
        "eta":  [0.025],
        # "w":  np.array([0.3,0.4,0.5,0.6])/u_w,
        "w":  np.linspace(0.00001,1,30),
        "T": np.linspace(0.00001,50,30)/u_temp,
        # "T": np.array([0.01, 15, 30, 50])/u_temp,
        "wd":  [5],
        "s": np.array([1,-1]),
        "m": [ np.array([0.85, 1.38]) ],
        "ef": [ np.array([290, 70]) ],
        "g": [ np.array([10, 0.001]), np.array([0.001, 10]),  np.array([10,5]), np.array([0.001, 0.001])],
        "pre_d0": np.array([0.3,0.7]),
        "v": [0.02, 0.4]
    }
]


params = []
for p in params_:
    for Ne in p["Ne"]:
        for wd in p["wd"]:
            for m in p["m"]:
                for ef in p["ef"]:
                    for v in p["v"]:
                        for g in p["g"]:
                            for w in p["w"]:
                                for T in p["T"]:
                                    for eta in p["eta"]:
                                        params.append({
                                                        "Ne": Ne,
                                                        "T": T,
                                                        "wd": wd,
                                                        "s": p["s"],
                                                        "m": m,
                                                        "ef": ef,
                                                        "g": g,
                                                        "pre_d0": p["pre_d0"],
                                                        "v": v,
                                                        "w":  w,
                                                        "eta": eta
                                                    })

print(len(params),'parameters generated')
