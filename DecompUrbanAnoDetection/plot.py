import json
import matplotlib.pyplot as plt

d = json.load(open("metrics/FAL-0-w2.json", "r"))
# plt.subplot(1, 2, 1)
# plt.plot(d["prec@k"])
# plt.subplot(1, 2, 2)
# plt.plot(d["recall@k"])
plt.plot(d["recall@k"], d["prec@k"])
plt.savefig("test.png")
